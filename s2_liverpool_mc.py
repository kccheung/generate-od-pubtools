import datetime

import geopandas as gpd
import numpy as np
import pandas as pd

from constants import LIVERPOOL_SHP, LIVERPOOL
from generate_od import generator
from utils import od_sanity_print, set_all_seeds, compute_od_metrics, plot_od_topk_gradient  # wherever you put them

if __name__ == "__main__":
    # Load Liverpool area
    area = gpd.read_file(LIVERPOOL_SHP)

    # Load baseline OD (generation.npy)
    baseline = np.load("./assets/example_data/CommutingOD/GB_Liverpool/generation.npy")

    results = []
    seeds = range(10)  # 10 runs
    dt_str = str(datetime.datetime.now()).replace(" ", "").replace(":", "").replace("-", "")[4:8]

    for seed in seeds:
        set_all_seeds(seed)

        # fresh generator per run (so weights, scalers etc. are reloaded cleanly)
        gen = generator.Generator()
        gen.city_name = LIVERPOOL
        gen.set_satetoken("xxxxxxxxxxxxxxx")
        gen.load_area(area)

        od_hat = gen.generate(sample_times=50)  # keep same as "main" run
        od_sanity_print(od_hat)

        metrics = compute_od_metrics(od_hat, baseline)
        metrics["seed"] = seed
        results.append(metrics)

        print(
            f"[Run seed={seed}] "
            f"Scaled total flows: {metrics['scaled_total']:.1f}, "
            f"RMSE: {metrics['rmse']:.3f}, "
            f"NRMSE: {metrics['nrmse']:.3f}, "
            f"CPC: {metrics['cpc']:.3f}"
        )

        # save the OD matrix to csv
        od_df = pd.DataFrame(
            od_hat,
            index=gen.area.index,  # or use area["region_id"] if you have
            columns=gen.area.index
        )
        od_df.to_csv(f"./outputs/seed_{seed}_od_matrix_liverpool_{dt_str}.csv")

        # save plot of top-k flows
        fig = plot_od_topk_gradient(od_hat, area, k=0.9, cmap_name="Reds")
        fig.savefig(f"./outputs/od_arc_liverpool_top0.9_50_samples_gpu_0.3a_imagex_{dt_str}.png", bbox_inches="tight", dpi=300)

    # Put into a DataFrame and compute mean ± std
    df = pd.DataFrame(results)
    print("\n=== Monte Carlo summary over seeds ===")
    print(df)

    summary = df[["rmse", "nrmse", "cpc"]].agg(["mean", "std"])
    print("\nMean ± std:")
    print(summary)
