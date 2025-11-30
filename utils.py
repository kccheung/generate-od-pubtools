# in utils.py (or wherever you keep it)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# fukuoka_population.py
import pandas as pd
import geopandas as gpd

from constants import FUKUOKA_WARD_STATS


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
