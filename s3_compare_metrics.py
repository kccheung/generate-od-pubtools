import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import od_sanity_print, rmse, nrmse, cpc
from constants import OD_PATH_LIVERPOOL


def main():
    """
    Verify reproduction by comparing the OD matrix pipeline generates for GB_Liverpool with
    the example generation.npy provided by the authors (MSE / correlation / visual pattern).
    """

    od_ref = np.load("./assets/example_data/CommutingOD/GB_Liverpool/generation.npy", allow_pickle=True)
    print(type(od_ref), getattr(od_ref, "shape", None))
    od_sanity_print(od_ref)
    print(od_ref)

    df = pd.read_csv(OD_PATH_LIVERPOOL, header=0, index_col=0)
    od_hat = df.to_numpy(dtype=float)
    print("OD_HAT shape (after dropping labels):", od_hat.shape)
    od_sanity_print(od_hat)
    print(od_hat)

    scale = od_ref.sum() / od_hat.sum()
    od_hat_scaled = od_hat * scale

    print("Scaled total flows:", od_hat_scaled.sum())
    print("RMSE (scaled):", rmse(od_ref, od_hat_scaled))
    print("NRMSE (scaled):", nrmse(od_ref, od_hat_scaled))
    print("CPC (scaled):", cpc(od_ref, od_hat_scaled))

    plt.figure(figsize=(5, 4))
    plt.imshow(od_ref, interpolation="nearest")
    plt.title("Reference Liverpool OD (generation.npy)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("./docs/img/liverpool_ref_od.png", dpi=200)

    plt.figure(figsize=(5, 4))
    plt.imshow(od_hat_scaled, interpolation="nearest")
    plt.title("Generated Liverpool OD (scaled)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("./docs/img/liverpool_hat_od.png", dpi=200)

    plt.figure(figsize=(5, 4))
    plt.imshow(od_hat_scaled - od_ref, interpolation="nearest")
    plt.title("Difference (generated - reference)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("./docs/img/liverpool_diff_od.png", dpi=200)


if __name__ == "__main__":
    main()

"""
<class 'numpy.ndarray'> (252, 252)
[[  0.   1.   2. ...   0.   0.   0.]
 [ 15.   0.   0. ...   0.   0.   0.]
 [ 15.   0.   0. ...   0.   0.   0.]
 ...
 [  3.   4.   5. ...   0.  92.  64.]
 [  1.   1.   3. ... 190.   0. 105.]
 [  4.   5.   9. ...  93.  78.   0.]]
"""
"""
vs. od_matrix_liverpool_2025-12-03_174537.224411
Scaled total flows: 4883625.0
RMSE (scaled): 74.63998360095168
NRMSE (scaled): 1.0175715743636948
CPC (scaled): 0.708976188958977
"""
"""
vs. od_liverpool_imageexport/od_matrix_liverpool_2025-12-02_203358.822592.csv
Scaled total flows: 4883625.0
RMSE (scaled): 74.44036127764443
NRMSE (scaled): 1.0148501107190633
CPC (scaled): 0.7028729071411474
"""
"""
vs. od_liverpool_imageexport/od_matrix_liverpool_2025-12-06_135455.481068.csv
Scaled total flows: 4883625.0
RMSE (scaled): 70.98601893429779
NRMSE (scaled): 0.9677568450572845
CPC (scaled): 0.7120679802093081
"""
