import numpy as np
import geopandas as gpd
import pandas as pd

from constants import OD_PATH, FUKUOKA_SHP, FUKUOKA_POPULATION_CSV
from utils import load_od_from, od_sanity_print, build_fukuoka_features_from_csv

od_path = OD_PATH
SHP_PATH = FUKUOKA_SHP


def main():
    print(f"Loading OD matrix from {od_path} ...")
    od_hat_fukuoka = load_od_from(od_path)  # shape (N, N)
    od = od_hat_fukuoka
    od_sanity_print(od)
    N = od.shape[0]

    # Outgoing commuters from each cell (origins)
    out_flows = od.sum(axis=1)  # shape (N,)

    # Incoming commuters to each cell (destinations)
    in_flows = od.sum(axis=0)   # shape (N,)

    print(f"Loading geometries from {SHP_PATH} ...")
    gdf_fukuoka = gpd.read_file(SHP_PATH)
    gdf = gdf_fukuoka.copy()
    gdf["out_flow"] = out_flows
    gdf["in_flow"] = in_flows

    # Compare flows to population (scatter + correlation)
    pop = build_fukuoka_features_from_csv(
        gdf,
        csv_path=FUKUOKA_POPULATION_CSV,
        ward_col="N03_005",
    )
    print(pop)



if __name__ == "__main__":
    main()

