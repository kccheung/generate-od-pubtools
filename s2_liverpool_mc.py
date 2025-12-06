import datetime
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from constants import LIVERPOOL_SHP, LIVERPOOL
from generate_od import generator
from utils import od_sanity_print, set_all_seeds, compute_od_metrics, plot_od_topk_gradient  # wherever you put them

k = 0.9
city_name = LIVERPOOL
shp_path = LIVERPOOL_SHP
baseline_path = "./assets/example_data/CommutingOD/GB_Liverpool/generation.npy"

if __name__ == "__main__":
    # Load Liverpool area
    area = gpd.read_file(shp_path)

    # Load baseline OD (generation.npy)
    baseline = np.load(baseline_path)

    for sample_times in [5, 10, 20, 50]:
        print(f"Baseline run for {sample_times} samples:")

        results = []
        seeds = range(10)  # 10 runs
        dt_str = str(datetime.datetime.now()).replace(" ", "").replace(":", "").replace("-", "")[4:12]

        for seed in seeds:
            set_all_seeds(seed)

            # fresh generator per run (so weights, scalers etc. are reloaded cleanly)
            gen = generator.Generator()
            gen.city_name = city_name
            gen.set_satetoken("xxxxxxxxxxxxxxx")
            gen.load_area(area)

            start_time = time.perf_counter()
            od_hat = gen.generate(sample_times=sample_times)  # keep same as "main" run
            end_time = time.perf_counter()
            runtime_sec = end_time - start_time
            od_sanity_print(od_hat)

            metrics = compute_od_metrics(od_hat, baseline)
            metrics["seed"] = seed
            metrics["runtime_sec"] = runtime_sec
            results.append(metrics)

            print(
                f"[Run seed={seed}] "
                f"Scaled total flows: {metrics['scaled_total']:.1f}, "
                f"RMSE: {metrics['rmse']:.3f}, "
                f"NRMSE: {metrics['nrmse']:.3f}, "
                f"CPC: {metrics['cpc']:.3f}"
                f"Runtime: {runtime_sec:.3f} sec"
            )

            # save the OD matrix to csv
            od_df = pd.DataFrame(
                od_hat,
                index=gen.area.index,  # or use area["region_id"] if you have
                columns=gen.area.index
            )
            cn = city_name.replace(" ", "_").lower()
            od_df.to_csv(f"./outputs/seed_{seed}_od_matrix_{cn}_{dt_str}.csv")

            # save plot of top-k flows
            fig = plot_od_topk_gradient(od_hat, area, k=k, cmap_name="Reds")
            fig.savefig(f"./outputs/od_arc_{cn}_top{k}_{sample_times}_samples_gpu_0.3a_imagex_{dt_str}.png", bbox_inches="tight", dpi=300)

        # Put into a DataFrame and compute mean ± std
        df = pd.DataFrame(results)
        print(f"\n=== Monte Carlo summary over seeds with sample_times={sample_times} ===")
        print(df)

        summary = df[["rmse", "nrmse", "cpc", "runtime_sec"]].agg(["mean", "std"])
        print("\nMean ± std:")
        print(summary)
