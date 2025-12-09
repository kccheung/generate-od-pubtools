# in utils.py (or wherever you keep it)
import os
import warnings
from io import BytesIO

import random
import numpy as np
import torch
import contextily as cx
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
# fukuoka_population.py
import pandas as pd
from PIL import Image
from matplotlib.collections import LineCollection
from shapely.geometry import LineString

from constants import FUKUOKA_WARD_STATS


def set_all_seeds(seed: int):
    print(f"\n=== Using seed {seed} ===")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # MPS is less deterministic, but still set it
        torch.manual_seed(seed)


def load_od_from(od_path: str) -> np.ndarray:
    """
    Load OD matrix CSV file.
    Assumes square matrix with no header or index column.
    Read with header + index, then drop them; support .npy arrays
    """
    ext = os.path.splitext(od_path)[1].lower()
    if ext == ".npy":  # repo .npy format
        arr = np.load(od_path, allow_pickle=False)
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            od = arr.astype(float)
        else:
            # try to coerce to 2D via pandas if possible (e.g., saved DataFrame values)
            try:
                df = pd.DataFrame(arr)
                od = df.to_numpy(dtype=float)
            except Exception:
                raise ValueError(f"Unsupported .npy contents in {od_path}")
    else:  # my stored CSV format
        df = pd.read_csv(od_path, header=0, index_col=0)
        od = df.to_numpy(dtype=float)
    return od


def plot_od_quantile_bands(od, geometries, add_basemap: bool = True):
    """
    Plot OD flows in three quantile bands on a 1x3 figure:

      Panel A: flows >= 95th percentile
      Panel B: 90th <= flows < 95th percentile
      Panel C: 75th <= flows < 90th percentile

    od          : (N, N) OD matrix (numpy array or DataFrame)
    geometries  : GeoDataFrame of N zones (CRS assumed WGS84 or already metric)
    """
    # ensure numpy array
    od_arr = np.asarray(od, dtype=float)
    n = od_arr.shape[0]
    assert od_arr.shape[1] == n, "OD matrix must be square."

    # project geometries to Web Mercator
    g = geometries.copy()
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g_3857 = g.to_crs(epsg=3857)
    centroids = g_3857.geometry.centroid

    # build all OD line records (ignore diagonal and non-positive flows)
    line_records = []
    for i in range(n):
        for j in range(n):
            f_ij = od_arr[i, j]
            if i == j or f_ij <= 0:
                continue
            line = LineString([centroids.iloc[i], centroids.iloc[j]])
            line_records.append({"flow": f_ij, "geometry": line})

    if not line_records:
        raise ValueError("No positive OD flows to plot.")

    lines_gdf = gpd.GeoDataFrame(line_records, geometry="geometry", crs=g_3857.crs)
    flows = lines_gdf["flow"].values

    # compute quantiles
    q95 = np.quantile(flows, 0.95)
    q90 = np.quantile(flows, 0.90)
    q75 = np.quantile(flows, 0.75)

    band_A = lines_gdf[lines_gdf["flow"] >= q95].copy()  # top 5%
    band_B = lines_gdf[(lines_gdf["flow"] >= q90) & (lines_gdf["flow"] < q95)].copy()
    band_C = lines_gdf[(lines_gdf["flow"] >= q75) & (lines_gdf["flow"] < q90)].copy()

    print(
        f"[OD quantile bands] "
        f"q75={q75:.1f}, q90={q90:.1f}, q95={q95:.1f}; "
        f"A={len(band_A)}, B={len(band_B)}, C={len(band_C)} edges."
    )

    # prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=400)
    titles = [
        f"Flows ≥ 95th percentile\n(flow ≥ {q95:.0f}), # edges: {len(band_A)}",
        f"90–95th percentile\n({q90:.0f} ≤ flow < {q95:.0f}), # edges: {len(band_B)}",
        f"75–90th percentile\n({q75:.0f} ≤ flow < {q90:.0f}), # edges: {len(band_C)}",
    ]
    bands = [band_A, band_B, band_C]

    # consistent extent
    minx, miny, maxx, maxy = g_3857.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05

    for ax, band, title in zip(axes, bands, titles):
        # base grid boundaries
        g_3857.boundary.plot(ax=ax, linewidth=0.3, color="grey", alpha=0.5)

        if len(band) > 0:
            flows_b = band["flow"].values

            # if there are at least a few edges, split into three sub-bands
            if len(flows_b) >= 3:
                b_q75 = np.quantile(flows_b, 0.75)
                b_q97 = np.quantile(flows_b, 0.97)

                band_low = band[band["flow"] <= b_q75].copy()
                band_mid = band[(band["flow"] > b_q75) & (band["flow"] <= b_q97)].copy()
                band_high = band[band["flow"] > b_q97].copy()

                # low flows within this band (blue, thin, faint) – same style as plot_od_arc_chart
                if len(band_low) > 0:
                    band_low.plot(
                        ax=ax,
                        linewidth=0.05,
                        color="#0308F8",  # blue
                        alpha=0.2,
                    )

                # medium flows within this band (red, medium)
                if len(band_mid) > 0:
                    band_mid.plot(
                        ax=ax,
                        linewidth=0.08,
                        color="#FD0B1B",  # red
                        alpha=0.5,
                    )

                # high flows within this band (yellow, thick, bright)
                if len(band_high) > 0:
                    band_high.plot(
                        ax=ax,
                        linewidth=0.12,
                        color="yellow",
                        alpha=0.8,
                    )
            else:
                # too few edges to split meaningfully; plot them all as medium strength
                band.plot(
                    ax=ax,
                    linewidth=0.08,
                    color="#FD0B1B",
                    alpha=0.5,
                )

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal")

        if add_basemap:
            try:
                cx.add_basemap(ax, crs=g_3857.crs, source=cx.providers.CartoDB.Positron)
            except Exception as e:
                warnings.warn(f"Could not add basemap: {e}")

        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["right", "top", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_od_ego_star(
        od,
        geometries,
        cbd_dict,
        top_k=50,
        direction="out",
        title_prefix="Top flows from",
):
    """
    Plot 'ego-network' OD maps for one or more origins (e.g. CBDs).

    For each entry in `cbd_dict`, make a separate figure showing
    the top_k strongest flows either:
      - outgoing FROM the CBD (direction="out"; OD row), or
      - incoming TO the CBD (direction="in"; OD column).

    Parameters
    ----------
    od : np.ndarray or (N,N) array-like
        OD matrix.
    geometries : GeoDataFrame
        N polygons for zones (CRS assumed WGS84 or already set).
    cbd_dict : dict[str, int]
        Mapping from CBD name to zone index, e.g.
        {"Tenjin": 293, "Hakata": 332}.
    top_k : int
        Number of strongest flows to draw per origin.
    direction : {"out", "in"}
        "out": flows from CBD to others (row of OD, plotted in Blues).
        "in" : flows from others into CBD (column of OD, plotted in Reds).
    title_prefix : str
        Prefix for axis title when direction="out".
    """
    od_arr = np.asarray(od, dtype=float)
    n = od_arr.shape[0]
    assert od_arr.shape == (n, n)

    # project once
    g = geometries.copy()
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g_3857 = g.to_crs(epsg=3857)
    centroids = g_3857.geometry.centroid

    figs = []
    for cbd_name, o in cbd_dict.items():
        if o < 0 or o >= n:
            raise ValueError(f"Origin index {o} (CBD '{cbd_name}') out of range 0..{n - 1}")

        if direction == "out":
            # Outgoing flows: row o
            flows_vec = od_arr[o, :].copy()
            flows_vec[o] = 0.0
            cmap = cm.get_cmap("Blues")
            cbar_label = "Outgoing flow (commuters)"
            title = f"{title_prefix} {cbd_name} (cell {o})\nTop {top_k} outgoing flows"
            origin_pt = centroids.iloc[o]

            def make_line(j):
                # CBD -> destination j
                return LineString([origin_pt, centroids.iloc[j]])
        elif direction == "in":
            # Incoming flows: column o
            flows_vec = od_arr[:, o].copy()
            flows_vec[o] = 0.0
            cmap = cm.get_cmap("Reds")
            cbar_label = "Incoming flow (commuters)"
            title = f"Top {top_k} incoming flows to {cbd_name} (cell {o})"
            origin_pt = centroids.iloc[o]

            def make_line(j):
                # origin j -> CBD
                return LineString([centroids.iloc[j], origin_pt])
        else:
            raise ValueError("direction must be 'out' or 'in'")

        pos_mask = flows_vec > 0
        idxs = np.where(pos_mask)[0]
        if len(idxs) == 0:
            print(
                f"[plot_od_ego_star] Origin {o} (CBD '{cbd_name}') "
                f"has no positive flows for direction='{direction}'."
            )
            continue

        # pick top_k
        k = min(top_k, len(idxs))
        top_idx = idxs[np.argpartition(-flows_vec[idxs], k - 1)[:k]]
        top_flows = flows_vec[top_idx]

        # sort so weak on bottom, strong on top
        order = np.argsort(top_flows)
        top_idx = top_idx[order]
        top_flows = top_flows[order]

        # build line segments
        line_geoms = [make_line(j) for j in top_idx]

        vmin, vmax = float(top_flows.min()), float(top_flows.max())
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # width ~ strength
        widths = 0.5 + 2.5 * (top_flows - vmin) / (vmax - vmin + 1e-9)

        lc = LineCollection(
            [np.array(line.coords) for line in line_geoms],
            array=top_flows,
            cmap=cmap,
            norm=norm,
            linewidths=widths,
            alpha=0.8,
        )

        fig, ax = plt.subplots(figsize=(6, 6), dpi=400)

        # base grid
        g_3857.boundary.plot(ax=ax, linewidth=0.3, color="grey", alpha=0.5)
        ax.add_collection(lc)

        # highlight origin cell
        g_3857.iloc[[o]].plot(
            ax=ax,
            facecolor="yellow",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.8,
            zorder=4,
        )

        minx, miny, maxx, maxy = g_3857.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        cx.add_basemap(ax, crs=g_3857.crs, source=cx.providers.CartoDB.Positron)

        ax.set_axis_off()
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(top_flows)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(cbar_label)

        ax.set_title(title)
        fig.tight_layout()
        figs.append(fig)

    return figs


def plot_od_topk_gradient(od, geometries, k=1000, cmap_name="Reds",
                          highlight_idxs=None, highlight_color="yellow",
                          highlight_alpha=0.25, highlight_edgecolor=None,
                          highlight_linewidth=0.8):
    """
    Plot top-k OD flows with a single gradient color and a colorbar legend.

    od          : (N, N) OD matrix
    geometries  : GeoDataFrame of N zones
    k           : number of strongest flows to draw
    cmap_name   : matplotlib colormap name, e.g. 'Blues', 'Reds', 'viridis'

    Optional highlighting parameters (for highlighting CBD cells etc.):
      highlight_idxs    : None | list[int] | array-like[bool]
                          If list/array of ints, treated as zone indices to highlight.
                          If boolean mask (len N) will be used directly.
      highlight_color   : fill color for highlighted polygons (default 'red')
      highlight_alpha   : transparency for highlighted polygons (default 0.25)
      highlight_edgecolor: edge color for highlighted polygons (defaults to same as fill)
      highlight_linewidth: edge line width for highlighted polygons (default 0.8)
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

    # 3. select top-k strongest flows (support k as count or top-fraction)
    if isinstance(k, float) and 0 < k < 1:  # fraction for decimal k in (0, 1)
        k_count = max(1, int(np.ceil(k * len(flows))))
    elif isinstance(k, (int, np.integer)) and k >= 1:  # top-k count
        k_count = min(int(k), len(flows))
    else:
        raise ValueError("`k` must be a positive integer or a float in (0,1) representing fraction")
    top_idx = np.argpartition(-flows, k_count - 1)[:k_count]

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

    # --- OPTIONAL: highlight selected polygons (e.g., CBD cells) ---
    if highlight_idxs is not None:
        # create boolean mask of length n
        try:
            mask = np.asarray(highlight_idxs)
            if mask.dtype == bool:
                if mask.shape[0] != n:
                    raise ValueError("Boolean highlight mask length must equal number of zones")
            else:
                # assume list/array of indices
                idxs = np.asarray(highlight_idxs, dtype=int)
                mask = np.zeros(n, dtype=bool)
                mask[idxs] = True
        except Exception:
            # fallback: try treating as pandas Series of booleans or ints
            try:
                import pandas as _pd
                if isinstance(highlight_idxs, (_pd.Series, _pd.Index)):
                    arr = np.asarray(highlight_idxs)
                    if arr.dtype == bool:
                        mask = arr
                    else:
                        idxs = arr.astype(int)
                        mask = np.zeros(n, dtype=bool)
                        mask[idxs] = True
                else:
                    raise
            except Exception:
                raise ValueError("Invalid `highlight_idxs` - must be boolean mask or list/array of indices")

        g_high = g[mask]
        if len(g_high) > 0:
            edgec = highlight_edgecolor if highlight_edgecolor is not None else highlight_color
            # plot highlighted polygons on top of basemap and lines
            g_high.plot(
                ax=ax,
                facecolor=highlight_color,
                edgecolor=edgec,
                linewidth=highlight_linewidth,
                alpha=highlight_alpha,
                zorder=4,
            )

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


def plot_population_heatmap(
        geometries: gpd.GeoDataFrame,
        pop_col: str = "pop",
        cmap_name: str = "viridis",
        per_km2: bool = False,
        title: str | None = None,
):
    """
    Plot a choropleth heatmap of population (or population density) over the given regions.
    """
    if pop_col not in geometries.columns:
        raise KeyError(f"Column '{pop_col}' not found in geometries.")

    # project to Web Mercator for nicer basemap overlay and area computation
    g = geometries.to_crs(epsg=3857).copy()

    if per_km2:
        # compute area in km^2
        area_km2 = g.geometry.area.values / 1e6
        density = g[pop_col].to_numpy(dtype=float) / (area_km2 + 1e-9)
        g["pop_density"] = density
        column_to_plot = "pop_density"
        legend_label = "Population density (per km²)"
        default_title = "Population density heatmap"
    else:
        column_to_plot = pop_col
        legend_label = "Population per cell"
        default_title = "Population heatmap"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=400)

    # base polygons colored by population (or density)
    g.plot(
        ax=ax,
        column=column_to_plot,
        cmap=cmap_name,
        linewidth=0.3,
        edgecolor="grey",
        alpha=0.5,
        legend=True,
        legend_kwds={"label": legend_label},
    )

    # add basemap
    cx.add_basemap(ax, crs=g.crs, source=cx.providers.CartoDB.Positron)

    ax.set_axis_off()
    ax.set_title(title if title is not None else default_title)

    return fig


def plot_flow_population_heatmaps(
        geometries: gpd.GeoDataFrame,
        pop_col: str = "pop",
        in_col: str = "in_flow",
        out_col: str = "out_flow",
        cmap_name: str = "Reds",
        highlight_idxs=None,
        gif_path: str | None = None,
        gif_duration: int = 2000,
):
    """
    Plot side-by-side heatmaps for:
      - population,
      - incoming flow,
      - outgoing flow,

    and report their correlations.

    Additionally:
      - highlight specific cells (e.g. CBDs) if `highlight_idxs` is provided,
      - optionally save a small GIF that flips between the three maps
        if `gif_path` is provided.

    Parameters
    ----------
    geometries : GeoDataFrame
        GeoDataFrame of regions with population and flow columns.
    pop_col : str, default "pop"
        Column for population per cell.
    in_col : str, default "in_flow"
        Column for total incoming commuters per cell.
    out_col : str, default "out_flow"
        Column for total outgoing commuters per cell.
    cmap_name : str, default "Reds"
        Matplotlib colormap name for the flow heatmaps.
    highlight_idxs : None | list[int] | array-like[bool]
        If list/array of ints, treated as zone indices to highlight.
        If boolean mask (len N), will be used directly.
    gif_path : str or None
        If not None, save an animated GIF at this path that flips between
        population, incoming flow, and outgoing flow maps.
    gif_duration : int
        Duration of each frame in the GIF in milliseconds.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object with the 1x3 subplot layout.
    """
    for col in [pop_col, in_col, out_col]:
        if col not in geometries.columns:
            raise KeyError(f"Column '{col}' not found in geometries.")

    # compute correlations (drop cells with NaN)
    df_corr = geometries[[pop_col, in_col, out_col]].dropna()
    corr_in = df_corr[[pop_col, in_col]].corr().iloc[0, 1]
    corr_out = df_corr[[pop_col, out_col]].corr().iloc[0, 1]
    print(f"[Flow vs population] Corr(pop, {in_col})  = {corr_in:.3f}")
    print(f"[Flow vs population] Corr(pop, {out_col}) = {corr_out:.3f}")

    # project to Web Mercator once
    g = geometries.to_crs(epsg=3857).copy()
    n = len(g)

    # build highlight mask if requested (similar semantics to plot_od_topk_gradient)
    highlight_mask = None
    if highlight_idxs is not None:
        try:
            arr = np.asarray(highlight_idxs)
            if arr.dtype == bool:
                if arr.shape[0] != n:
                    raise ValueError("Boolean highlight mask length must equal number of zones")
                highlight_mask = arr
            else:
                idxs = arr.astype(int)
                mask = np.zeros(n, dtype=bool)
                mask[idxs] = True
                highlight_mask = mask
        except Exception as e:
            raise ValueError(
                "Invalid `highlight_idxs` - must be boolean mask or list/array of indices"
            ) from e

    if highlight_mask is not None:
        g_high = g[highlight_mask]
        # precompute centroids for marker placement
        cbd_centroids = g_high.geometry.centroid
    else:
        g_high = None
        cbd_centroids = None

    # --- main 1x3 static figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=400)
    ax_pop, ax_in, ax_out = axes

    # 1) Population
    g.plot(
        ax=ax_pop,
        column=pop_col,
        cmap="viridis",
        linewidth=0.3,
        edgecolor="grey",
        alpha=0.5,
        legend=True,
        legend_kwds={"label": "Population per cell"},
    )
    cx.add_basemap(ax_pop, crs=g.crs, source=cx.providers.CartoDB.Positron)
    if cbd_centroids is not None and len(cbd_centroids) > 0:
        ax_pop.scatter(
            cbd_centroids.x,
            cbd_centroids.y,
            marker="*",
            s=10,
            edgecolors="black",
            facecolors="yellow",
            linewidths=0.1,
            zorder=5,
        )
    ax_pop.set_axis_off()
    ax_pop.set_title("Population")

    # 2) Incoming flow
    g.plot(
        ax=ax_in,
        column=in_col,
        cmap=cmap_name,
        linewidth=0.3,
        alpha=0.5,
        edgecolor="grey",
        legend=True,
        legend_kwds={"label": "Incoming commuters"},
    )
    cx.add_basemap(ax_in, crs=g.crs, source=cx.providers.CartoDB.Positron)
    if cbd_centroids is not None and len(cbd_centroids) > 0:
        ax_in.scatter(
            cbd_centroids.x,
            cbd_centroids.y,
            marker="*",
            s=10,
            edgecolors="black",
            facecolors="yellow",
            linewidths=0.1,
            zorder=5,
        )
    ax_in.set_axis_off()
    ax_in.set_title(f"Incoming flow\nCorr(pop, in) = {corr_in:.2f}")

    # 3) Outgoing flow
    g.plot(
        ax=ax_out,
        column=out_col,
        cmap="Blues",
        linewidth=0.3,
        alpha=0.5,
        edgecolor="grey",
        legend=True,
        legend_kwds={"label": "Outgoing commuters"},
    )
    cx.add_basemap(ax_out, crs=g.crs, source=cx.providers.CartoDB.Positron)
    if cbd_centroids is not None and len(cbd_centroids) > 0:
        ax_out.scatter(
            cbd_centroids.x,
            cbd_centroids.y,
            marker="*",
            s=10,
            edgecolors="black",
            facecolors="yellow",
            linewidths=0.1,
            zorder=5,
        )
    ax_out.set_axis_off()
    ax_out.set_title(f"Outgoing flow\nCorr(pop, out) = {corr_out:.2f}")

    fig.tight_layout()

    # --- optional GIF generation: flip between the three maps ---
    if gif_path is not None:
        from io import BytesIO

        def fig_to_image(single_fig):
            buf = BytesIO()
            single_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            buf.seek(0)
            return Image.open(buf)

        frames = []

        # helper to draw a single map type on its own figure
        def make_single_map(column, title, cmap):
            f, ax = plt.subplots(figsize=(6, 6), dpi=300)
            g.plot(
                ax=ax,
                column=column,
                cmap=cmap,
                linewidth=0.3,
                edgecolor="grey",
                alpha=0.5,
                legend=False,
            )
            cx.add_basemap(ax, crs=g.crs, source=cx.providers.CartoDB.Positron)
            if cbd_centroids is not None and len(cbd_centroids) > 0:
                ax.scatter(
                    cbd_centroids.x,
                    cbd_centroids.y,
                    marker="*",
                    s=10,
                    edgecolors="black",
                    facecolors="yellow",
                    linewidths=0.1,
                    zorder=5,
                )
            ax.set_axis_off()
            ax.set_title(title)
            img = fig_to_image(f)
            plt.close(f)
            return img

        frames.append(make_single_map(pop_col, "Population", "viridis"))
        frames.append(make_single_map(in_col, "Incoming flow", cmap_name))
        frames.append(make_single_map(out_col, "Outgoing flow", "Blues"))

        if len(frames) > 0:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=gif_duration,
                loop=0,
            )

    return fig


def plot_od_diff_arcs(
    od_ref,
    od_hat,
    geometries,
    k: int = 1000,
    add_basemap: bool = True,
    title: str | None = None,
):
    """
    Plot a 'difference arcs' map between reference and generated OD matrices.

    - od_ref: reference OD (e.g. generation.npy from repo)
    - od_hat: your reproduced / generated OD (scaled to same total if desired)
    - geometries: GeoDataFrame of N zones (polygons)
    - k: keep top-k OD pairs by absolute difference |od_hat - od_ref|
    - add_basemap: whether to draw a web basemap under the arcs
    """
    od_ref_arr = np.asarray(od_ref, dtype=float)
    od_hat_arr = np.asarray(od_hat, dtype=float)
    assert od_ref_arr.shape == od_hat_arr.shape, "Reference and generated OD must have same shape."
    n = od_ref_arr.shape[0]
    assert od_ref_arr.shape[1] == n, "OD must be square."

    # difference matrix (generated - reference)
    diff = od_hat_arr - od_ref_arr
    abs_diff = np.abs(diff)

    # flatten and pick top-k by |difference|
    abs_flat = abs_diff.ravel()
    pos_mask = abs_flat > 0
    idx_all = np.where(pos_mask)[0]
    if len(idx_all) == 0:
        raise ValueError("No non-zero differences between OD matrices.")

    k_eff = min(k, len(idx_all))
    top_flat_idx = idx_all[np.argpartition(-abs_flat[idx_all], k_eff - 1)[:k_eff]]
    top_vals = diff.ravel()[top_flat_idx]
    top_abs = abs_flat[top_flat_idx]

    # sort by absolute difference so weakest arcs are drawn first, strongest last
    order = np.argsort(top_abs)  # ascending |diff|
    top_flat_idx = top_flat_idx[order]
    top_vals = top_vals[order]
    top_abs = top_abs[order]

    # map flat indices -> (i, j)
    ij = [divmod(idx, n) for idx in top_flat_idx]
    is_idx = [p[0] for p in ij]
    js_idx = [p[1] for p in ij]

    # project geometries once
    g = geometries.copy()
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g_3857 = g.to_crs(epsg=3857)
    centroids = g_3857.geometry.centroid

    # build line geometries
    line_geoms = [
        LineString([centroids.iloc[i], centroids.iloc[j]])
        for i, j in zip(is_idx, js_idx)
    ]

    # norm centered at zero for diverging colours
    vmax = float(top_abs.max())
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cm.get_cmap("bwr")  # blue = under, red = over

    # line widths scaled by |difference|
    widths = 0.5 + 2.5 * (top_abs / (top_abs.max() + 1e-9))

    lc = LineCollection(
        [np.array(line.coords) for line in line_geoms],
        array=top_vals,
        cmap=cmap,
        norm=norm,
        linewidths=widths,
        alpha=0.4,
    )

    fig, ax = plt.subplots(figsize=(8, 8), dpi=400)

    # base grid
    g_3857.boundary.plot(ax=ax, linewidth=0.3, color="grey", alpha=0.5)
    ax.add_collection(lc)

    # extent
    minx, miny, maxx, maxy = g_3857.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")

    if add_basemap:
        try:
            cx.add_basemap(ax, crs=g_3857.crs, source=cx.providers.CartoDB.Positron)
        except Exception as e:
            warnings.warn(f"Could not add basemap: {e}")

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["right", "top", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    if title is None:
        title = f"Top {k_eff} OD differences (generated - reference)"
    ax.set_title(title)

    # colourbar legend
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(top_vals)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("OD difference (generated - reference)")
    # Plot arcs with diverging colour (blue: under, red: over my hat).

    fig.tight_layout()
    return fig


def rmse(F, F_hat):
    diff = F - F_hat
    return np.sqrt(np.mean(diff ** 2))


def nrmse(F, F_hat):
    # as in their dataset paper: normalize by variance around mean of true F
    F_mean = F.mean()
    denom = np.sqrt(np.mean((F - F_mean) ** 2))
    return rmse(F, F_hat) / denom


def cpc(F, F_hat):
    # Common Part of Commuting
    # CPC = \frac{2 \times \text{overlap}}{\text{true total} + \text{predicted total}}
    numerator = 2 * np.sum(np.minimum(F, F_hat))
    denominator = np.sum(F) + np.sum(F_hat)
    return numerator / denominator


def compute_od_metrics(od_hat: np.ndarray, baseline: np.ndarray):
    """
    Compare generated OD with Liverpool baseline.
    Returns dict with scaled_total, rmse, nrmse, cpc.
    """
    assert od_hat.shape == baseline.shape

    # Work in float, keep original 2D shape
    F = baseline.astype(float)
    F_hat = od_hat.astype(float)

    # Scale F_hat so that total commuters matches baseline total
    total_F = F.sum()
    total_F_hat = F_hat.sum()
    scale = total_F / total_F_hat if total_F_hat > 0 else 1.0
    F_hat_scaled = F_hat * scale

    # Use the helper functions defined above so formulas are identical
    rmse_val = rmse(F, F_hat_scaled)
    nrmse_val = nrmse(F, F_hat_scaled)
    cpc_val = cpc(F, F_hat_scaled)

    return {
        "scaled_total": F_hat_scaled.sum(),
        "rmse": rmse_val,
        "nrmse": nrmse_val,
        "cpc": cpc_val,
    }


def get_fukuoka_cbd_idxs(gdf):
    from shapely.geometry import Point

    # CBD points dictionary
    cbd_points = {
        "Hakata": Point(130.4207, 33.5899),  # 33.5899062199416, 130.42066302391717
        "Tenjin": Point(130.3988, 33.5916),  # 33.591570379326306, 130.39883588158744
        "Nishijin": Point(130.3566, 33.5835),  # 33.5835205, 130.3566309
        "Ohashi": Point(130.4267, 33.5593),  # 33.55926823092667, 130.4266945111465
        "Yakuin": Point(130.4022, 33.5821)  # 33.58214140454225, 130.40216137738247
    }

    gdf_m = gdf.to_crs(epsg=32652)  # UTM zone 52N for Kyushu region

    # Project points
    cbd_points_m = {}
    for name, pt in cbd_points.items():
        cbd_points_m[name] = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(32652).iloc[0]

    def nearest_cell_idx_proj(gdf_m, point_m):
        distances = gdf_m.geometry.centroid.distance(point_m)
        return distances.idxmin()

    # Find indices
    cbd_idxs = {}
    for name, pt_m in cbd_points_m.items():
        cbd_idxs[name] = nearest_cell_idx_proj(gdf_m, pt_m)

    # only for s4 script
    if "in_flow" in gdf:
        gdf["in_rank"] = gdf["in_flow"].rank(ascending=False, method="min")
        gdf["out_rank"] = gdf["out_flow"].rank(ascending=False, method="min")
        data = []
        for name, idx in cbd_idxs.items():
            data.append({
                "Name": name,
                "Cell Index": idx,
                "In Flow": gdf.loc[idx, "in_flow"],
                "Out Flow": gdf.loc[idx, "out_flow"],
                "Population": gdf.loc[idx, "pop"],
                "In Rank": gdf.loc[idx, "in_rank"],
                "Out Rank": gdf.loc[idx, "out_rank"]
            })
        table = pd.DataFrame(data).sort_values(by="Population", ascending=False)
        print(table.to_string(index=False))

    # Return indices in original order
    # return list(cbd_idxs.values())
    return cbd_idxs


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


def build_fukuoka_features_from_csv(
        area_gdf: gpd.GeoDataFrame,
        csv_path: str,
        ward_col: str = "N03_005",
) -> np.ndarray:
    """
    NOT USED ANYMORE, kept for reference, use 100m WorldPop TIF instead.

    Build [population, area_km2] features for each grid cell in `area_gdf`
    using Fukuoka City's ward-level registered population CSV.

    Assumes:
      - `area_gdf` is a grid over Fukuoka-shi (e.g. fukuoka_shi_grid_300.shp),
      - it has a column `ward_col` (default "N03_005") with ward names like "東区", "博多区", ...,
      - `csv_path` points to the ward-level population CSV previously used.

    For each ward:
      - total ward population is read from the CSV (Japanese + foreign residents),
      - that population is distributed *equally* across all grid cells whose
        `ward_col` matches that ward,
      - each cell's area_km2 is computed from its polygon geometry.

    Returns
    -------
    features : np.ndarray, shape (N, 2)
        Column 0: population per cell
        Column 1: area_km2 per cell
    """
    if ward_col not in area_gdf.columns:
        raise KeyError(f"Column '{ward_col}' not found in area_gdf; "
                       "cannot map grid cells to ward-level population.")

    # 1. Load ward-level total populations from CSV
    ward_pops = load_fukuoka_ward_population_from_csv(csv_path)
    if not ward_pops:
        raise ValueError("No ward populations loaded from CSV; "
                         "check csv_path and CSV format.")

    # 2. Compute each cell's area in km^2 using a metric CRS
    area_metric = area_gdf.to_crs(epsg=3857).copy()
    cell_area_km2 = area_metric.geometry.area.values / 1e6  # m^2 -> km^2

    n = len(area_gdf)
    features = np.zeros((n, 2), dtype=float)

    # 3. Distribute each ward's population equally across its grid cells
    ward_series = area_gdf[ward_col].astype(str)
    ward_counts = ward_series.value_counts().to_dict()

    for i, ward_name in enumerate(ward_series):
        # Population for this ward (0 if not found)
        ward_pop_total = float(ward_pops.get(ward_name, 0.0))
        count = ward_counts.get(ward_name, 0)

        if count > 0 and ward_pop_total > 0:
            per_cell_pop = ward_pop_total / count
        else:
            per_cell_pop = 0.0  # e.g. sea / outside wards / missing ward

        features[i, 0] = per_cell_pop
        features[i, 1] = cell_area_km2[i]

    return features


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
