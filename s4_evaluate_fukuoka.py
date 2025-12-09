import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from constants import OD_PATH, FUKUOKA_SHP, FUKUOKA_POPULATION_CSV, JPN_TIF_PATH
from generate_od.worldpop import worldpop_from_local_tif
from utils import load_od_from, od_sanity_print, build_fukuoka_features_from_csv, plot_population_heatmap, plot_flow_population_heatmaps, get_fukuoka_cbd_idxs

od_path = OD_PATH
SHP_PATH = FUKUOKA_SHP


def main():
    print(f"Loading OD matrix from {od_path} ...")
    od_hat_fukuoka = load_od_from(od_path)  # shape (N, N)
    od = od_hat_fukuoka
    od_sanity_print(od)
    # N = od.shape[0]

    # Outgoing commuters from each cell (origins)
    out_flows = od.sum(axis=1)  # shape (N,)

    # Incoming commuters to each cell (destinations)
    in_flows = od.sum(axis=0)  # shape (N,)

    print(f"Loading geometries from {SHP_PATH} ...")
    gdf_fukuoka = gpd.read_file(SHP_PATH)
    gdf = gdf_fukuoka.copy()
    gdf["out_flow"] = out_flows
    gdf["in_flow"] = in_flows

    # Compare flows to population (scatter + correlation)
    worldpop_feats = worldpop_from_local_tif(gdf_fukuoka, JPN_TIF_PATH)
    # print(f"worldpop_feats: {worldpop_feats}")
    gdf["pop"] = worldpop_feats[:, 0]  # population count
    # below per-capita values show which cells are "over-serving" as job centres or sleeper suburbs
    gdf["out_per_capita"] = gdf["out_flow"] / (gdf["pop"] + 1e-6)  # add to avoid div by zero
    gdf["in_per_capita"] = gdf["in_flow"] / (gdf["pop"] + 1e-6)

    # Basic descriptive stats
    print(gdf[["pop", "out_flow", "in_flow", "out_per_capita", "in_per_capita", ]].describe().round())

    # check for CBD alignment (Tenjin, Hakata...)
    cbd_idxs = list(get_fukuoka_cbd_idxs(gdf))

    # population heatmap
    fig_pop = plot_population_heatmap(gdf, per_km2=False)
    fig_pop.savefig("./docs/img/fukuoka_pop_heatmap.png", dpi=200)

    # flow vs population heatmaps
    fig_flows = plot_flow_population_heatmaps(gdf, pop_col="pop",
                                              in_col="in_flow",
                                              out_col="out_flow",
                                              highlight_idxs=cbd_idxs,
                                              gif_path="./docs/img/fukuoka_pop_flow_heatmap.gif", )
    fig_flows.savefig("./docs/img/fukuoka_pop_flow_heatmaps.png", dpi=200)

    # Correlations
    corr_out = gdf[["pop", "out_flow"]].corr().iloc[0, 1]
    corr_in = gdf[["pop", "in_flow"]].corr().iloc[0, 1]
    print(f"Corr(pop, out_flow) = {corr_out:.3f}")
    print(f"Corr(pop, in_flow)  = {corr_in:.3f}")

    # Visualize
    plt.figure()
    plt.scatter(gdf["pop"], gdf["out_flow"], alpha=0.5)
    plt.xlabel("Population per cell")
    plt.ylabel("Outgoing commuters")
    plt.title("Population vs Outgoing flow (Fukuoka)")
    plt.tight_layout()
    plt.savefig("./docs/img/fukuoka_pop_vs_out.png", dpi=200)

    plt.figure()
    plt.scatter(gdf["pop"], gdf["in_flow"], alpha=0.5)
    plt.xlabel("Population per cell")
    plt.ylabel("Incoming commuters")
    plt.title("Population vs Incoming flow (Fukuoka)")
    plt.tight_layout()
    plt.savefig("./docs/img/fukuoka_pop_vs_in.png", dpi=200)


if __name__ == "__main__":
    main()
