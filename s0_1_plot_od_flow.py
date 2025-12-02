# s0_1_plot_od_flow.py

import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from constants import OD_PATH, FUKUOKA_SHP, LIVERPOOL_SHP  # same as in s1_sate_img_process.py
from generate_od.utils import plot_od_arc_chart
from utils import od_sanity_print, plot_od_topk_gradient


# SHP_PATH = LIVERPOOL_SHP
SHP_PATH = FUKUOKA_SHP
# liverpool od quantiles
LOW = 133
HIGH = 217


def main():
    parser = argparse.ArgumentParser(
        description="Plot OD flows as arc chart for a given city."
    )
    parser.add_argument(
        "--od_csv",
        type=str,
        # required=True,
        default=OD_PATH,
        help="Path to OD matrix CSV file (square matrix, no header).",
    )
    parser.add_argument(
        "--shapefile",
        type=str,
        default=SHP_PATH,
        help="Path to region shapefile / GeoPackage (default: constants.out_path).",
    )
    # parser.add_argument(
    #     "--low",
    #     type=float,
    #     default=LOW,
    #     help="Lower quantile for color scaling (e.g. 0.5).",
    # )
    # parser.add_argument(
    #     "--high",
    #     type=float,
    #     default=HIGH,
    #     help="Upper quantile for color scaling (e.g. 0.99).",
    # )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save figure (e.g. 'figures/fukuoka_od.png').",
    )

    args = parser.parse_args()

    # 1. Load OD matrix
    print(f"Loading OD matrix from {args.od_csv} ...")

    # <-- NEW: read with header + index, then drop them
    df = pd.read_csv(args.od_csv, header=0, index_col=0)
    od = df.to_numpy(dtype=float)
    print("OD shape (after dropping labels):", od.shape)
    od_sanity_print(od)

    print(f"Loading geometries from {args.shapefile} ...")
    gdf = gpd.read_file(args.shapefile)
    geometries = gdf.geometry
    n_regions = len(geometries)
    print("Number of regions in geometry:", n_regions)

    if od.shape != (n_regions, n_regions):
        raise ValueError(
            f"Mismatch: OD has shape {od.shape} but geometries has {n_regions} features."
        )

    # 3. Plot OD flows

    flows = od[od > 0].ravel()
    for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
        print(q, np.quantile(flows, q))
    q_low = np.quantile(flows, 0.95)
    q_high = np.quantile(flows, 0.999)
    print(f"Plotting OD arcs with low={q_low}, high={q_high} ...")
    # your plot function should accept an existing axis, or you can let it create inside
    # fig = plot_od_arc_chart(
    #     od,
    #     geometries,
    #     low=q_low,
    #     high=q_high,
    # )
    fig = plot_od_topk_gradient(od, gdf, k=200, cmap_name="Reds")
    # fig.savefig("od_fukuoka_top1000_blues.png", bbox_inches="tight", dpi=200)

    if args.output:
        print(f"Saving figure to {args.output} ...")
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
