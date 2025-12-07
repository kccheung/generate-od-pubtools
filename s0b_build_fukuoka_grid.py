# A script that builds a ~300-zone grid from current 950 sub-wards Fukuoka shp

import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box

# ---- settings ----
INPUT_SHP = "./assets/fukuoka_wards_n03b.shp"  # your 950-subward file
OUTPUT_SHP = "./assets/fukuoka_shi_grid_431_v3.shp"  # new coarse grid
TARGET_ZONES = 300  # approx number of cells
WARD_COL = "N03_005"  # ward name/code column in the input shapefile (e.g. 東区, 博多区, etc.)


def main():
    # 1. Load original 950-zone shapefile
    gdf = gpd.read_file(INPUT_SHP)
    if gdf.crs is None:
        # assume WGS84 if missing
        gdf = gdf.set_crs(epsg=4326)

    # 2. Project to metric CRS for area & grid construction
    gdf_m = gdf.to_crs(epsg=3857)

    # union to get whole Fukuoka city shape
    city_union = gdf_m.unary_union
    total_area_m2 = city_union.area
    print(f"Total area (km^2): {total_area_m2 / 1e6:.2f}")

    # 3. Decide grid cell size from target zone count
    #    cell_area ≈ total_area / TARGET_ZONES → cell_size = sqrt(area)
    cell_area_m2 = total_area_m2 / TARGET_ZONES
    cell_size = cell_area_m2 ** 0.5
    print(f"Target cell length (km): {cell_size / 1e3:.2f}")

    minx, miny, maxx, maxy = city_union.bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    cells = []
    for x in xs:
        for y in ys:
            geom = box(x, y, x + cell_size, y + cell_size)
            # only keep cells that intersect the city polygon
            if geom.intersects(city_union):
                cells.append(geom)

    grid_m = gpd.GeoDataFrame({"geometry": cells}, crs=gdf_m.crs)
    grid_m["zone_id"] = range(len(grid_m))

    # 3b. Attach ward information to each grid cell via centroid-based spatial join
    if WARD_COL in gdf_m.columns:
        print(f"Attaching ward info using column '{WARD_COL}'...")
        wards = gdf_m[[WARD_COL, "geometry"]].copy()
        # use centroids of grid cells for a robust point-in-polygon join
        grid_centroids = grid_m.copy()
        grid_centroids["geometry"] = grid_m.geometry.centroid

        joined = gpd.sjoin(
            grid_centroids,
            wards,
            how="left",
            predicate="within",  # centroid within ward polygon
        )
        # propagate ward label back to original grid_m
        grid_m[WARD_COL] = joined[WARD_COL].values
    else:
        print(f"Warning: ward column '{WARD_COL}' not found in INPUT_SHP; ward info will not be saved.")

    # cells with ward label
    has_ward = grid_m[WARD_COL].notna()
    print("Cells with ward:", has_ward.sum())
    print("Cells without ward:", (~has_ward).sum())

    # drop cells without ward assignment
    grid_m = grid_m[has_ward].reset_index(drop=True)
    grid_m["zone_id"] = range(len(grid_m))  # reassign IDs

    print(f"Number of grid cells created: {len(grid_m)}")

    # 4. Back to WGS84 for consistency with WorldPop / Esri code
    grid = grid_m.to_crs(epsg=4326)

    os.makedirs(os.path.dirname(OUTPUT_SHP), exist_ok=True)
    grid.to_file(OUTPUT_SHP)
    print(f"Saved grid shapefile to: {OUTPUT_SHP}")


if __name__ == "__main__":
    main()
