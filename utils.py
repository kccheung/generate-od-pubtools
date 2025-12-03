# in utils.py (or wherever you keep it)
from io import BytesIO

import contextily as cx
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
# fukuoka_population.py
import pandas as pd
from PIL import Image
from matplotlib.collections import LineCollection
from shapely.geometry import LineString

from constants import FUKUOKA_WARD_STATS


def plot_od_topk_gradient(od, geometries, k=1000, cmap_name="Blues"):
    """
    Plot top-k OD flows with a single gradient color and a colorbar legend.

    od          : (N, N) OD matrix
    geometries  : GeoDataFrame of N zones
    k           : number of strongest flows to draw
    cmap_name   : matplotlib colormap name, e.g. 'Blues', 'Reds', 'viridis'
    """
    # 1. project to Web Mercator
    g = geometries.to_crs(epsg=3857).copy()
    centroids = g.geometry.centroid

    n = od.shape[0]
    assert od.shape[0] == od.shape[1]

    # 2. flatten OD and ignore diagonal
    i_idx, j_idx = np.where(~np.eye(n, dtype=bool))
    flows = od[i_idx, j_idx]

    # only keep positive flows
    mask_pos = flows > 0
    i_idx = i_idx[mask_pos]
    j_idx = j_idx[mask_pos]
    flows = flows[mask_pos]

    if len(flows) == 0:
        raise ValueError("No positive flows to plot.")

    # 3. select top-k strongest flows
    k = min(k, len(flows))
    top_idx = np.argpartition(-flows, k - 1)[:k]

    # take the corresponding indices / flows
    i_top = i_idx[top_idx]
    j_top = j_idx[top_idx]
    f_top = flows[top_idx]

    # sort by flow ascending so weakest are drawn first, strongest last (on top)
    order = np.argsort(f_top)  # ascending
    i_top = i_top[order]
    j_top = j_top[order]
    f_top = f_top[order]

    # 4. build LineStrings in this sorted order
    line_geoms = [
        LineString([centroids.iloc[i], centroids.iloc[j]])
        for i, j in zip(i_top, j_top)
    ]

    # 5. set up colormap + normalisation based on actual OD values
    vmin, vmax = float(f_top.min()), float(f_top.max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    # line widths (optional: slightly thicker for stronger flows)
    widths = 0.3 + 2.7 * (f_top - vmin) / (vmax - vmin + 1e-9)

    # 6. create LineCollection; array=f_top so colorbar reflects OD values
    lc = LineCollection(
        [np.array(line.coords) for line in line_geoms],
        array=f_top,
        cmap=cmap,
        norm=norm,
        linewidths=widths,
        alpha=0.8,
    )

    # 7. plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=400)

    # base polygons as light outline
    g.boundary.plot(ax=ax, linewidth=0.3, color="grey", alpha=0.5)

    ax.add_collection(lc)

    # set limits from geometries
    minx, miny, maxx, maxy = g.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # add basemap
    cx.add_basemap(ax, crs=g.crs, source=cx.providers.CartoDB.Positron)
    ax.set_axis_off()

    # 8. colorbar legend
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(f_top)  # so the colorbar is tied to the actual OD values
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Predicted flow (commuters)")
    # format ticks as integers for readability
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{int(round(x))}"))

    ax.set_title(f"Top {k} OD flows (gradient {cmap_name})")

    return fig


def rmse(F, F_hat):
    diff = F - F_hat
    return np.sqrt(np.mean(diff**2))


def nrmse(F, F_hat):
    # as in their dataset paper: normalize by variance around mean of true F
    F_mean = F.mean()
    denom = np.sqrt(np.mean((F - F_mean)**2))
    return rmse(F, F_hat) / denom


def cpc(F, F_hat):
    # Common Part of Commuting
    numerator = 2 * np.sum(np.minimum(F, F_hat))
    denominator = np.sum(F) + np.sum(F_hat)
    return numerator / denominator


def load_fukuoka_ward_population_from_csv(csv_path: str) -> dict:
    """
    Read Fukuoka City's registered population CSV and return
    a dict: { ward_name (e.g. '東区'): total_population }.

    Total = 日本人_区 + 外国人_区.
    """
    df = pd.read_csv(csv_path)

    # pick rows for ward-level Japanese + foreigner
    wards_rows = df[df["区分"].str.contains("東区|博多区|中央区|南区|城南区|早良区|西区")]

    ward_totals = {}
    for _, row in wards_rows.iterrows():
        kind, ward = row["区分"].split("_", 1)  # e.g. '日本人', '東区'
        if not ward.endswith("区"):
            continue  # skip うち入部出張所 etc.
        ward_totals.setdefault(ward, 0)
        ward_totals[ward] += int(row["総人口"])

    return ward_totals


def build_fukuoka_features_from_csv(area_gdf: gpd.GeoDataFrame,
                                    csv_path: str,
                                    ward_col: str = "N03_005"):
    """
    Given a ward-level GeoDataFrame (N03),
    and the Fukuoka ward population CSV,
    return a (N, 2) array: [population_count, area_km2] per ward.
    """
    ward_pops = load_fukuoka_ward_population_from_csv(csv_path)

    # compute area in km^2 from geometry
    # (work in metric CRS, then convert to km^2)
    area_metric = area_gdf.to_crs(epsg=3857).copy()
    # area_km2 = area_metric.geometry.area.values / 1e6

    pops = []
    for idx, row in area_gdf.iterrows():
        ward_name = row[ward_col]  # e.g. '東区'
        if ward_name not in ward_pops:
            raise KeyError(f"Ward '{ward_name}' not found in CSV-derived ward_pops")

        pops.append([ward_pops[ward_name], FUKUOKA_WARD_STATS[ward_name]["area_km2"]])

    return np.asarray(pops, dtype=float)


def od_sanity_print(od_hat):
    print("OD matrix shape:", od_hat.shape)
    print("OD matrix (top-left 5x5):")
    print(od_hat[:5, :5])
    print("Min / max OD:", od_hat.min(), od_hat.max())
    print("Total flows:", od_hat.sum())
    print("Zero diagonal? ", (od_hat.diagonal() == 0).all())


def population_sanity_print(worldpop):
    # --- WorldPop population sanity check ---
    total_pop = float(worldpop[:, 0].sum())
    n_cells = len(worldpop[:, 0])
    print(f"[WorldPop sanity] Total population over {n_cells} cells: {total_pop:,.0f}")


def show_regional_image(regional_images, idx, high_res=False):
    """
    regional_images: dict[int, BytesIO or PIL.Image or np.ndarray]
    idx: region index key
    """
    img_obj = regional_images[idx]
    # img_obj = regional_images

    # 1) Convert to a PIL image
    if isinstance(img_obj, BytesIO):
        img_pil = Image.open(img_obj)
        print(img_pil.size)  # (width, height) in pixels, debug use
    elif isinstance(img_obj, Image.Image):
        img_pil = img_obj
    else:
        # maybe already a numpy array
        img_arr = np.array(img_obj)
        plt.imshow(img_arr)
        plt.axis("off")
        plt.title(f"Region {idx}")
        plt.show()
        return

    # 2) Convert to numpy for debugging / plotting
    img_arr = np.array(img_pil)

    # 3) Show
    if high_res:
        plt.figure(figsize=(10, 10))  # bigger window
    plt.imshow(img_arr)
    plt.axis("off")
    plt.title(f"Region {idx}")
    plt.show()
