import numpy as np
import pandas as pd

from utils import od_sanity_print
from constants import OD_PATH_LIVERPOOL

"""
Verify our reproduction by comparing the OD matrix our pipeline generates for GB_Liverpool with 
the example generation.npy provided by the authors (MSE / correlation / visual pattern).

TODO: Then for Fukuoka, generate a new OD matrix using the same pipeline and discuss SDG relevance etc.
"""

od_ref = np.load("./assets/example_data/CommutingOD/GB_Liverpool/generation.npy", allow_pickle=True)
print(type(od_ref), getattr(od_ref, "shape", None))
od_sanity_print(od_ref)
print(od_ref)

df = pd.read_csv(OD_PATH_LIVERPOOL, header=0, index_col=0)
od_hat = df.to_numpy(dtype=float)
print("OD shape (after dropping labels):", od_hat.shape)
od_sanity_print(od_hat)
print(od_hat)

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
