import numpy as np

"""
Verify our reproduction by comparing the OD matrix our pipeline generates for GB_Liverpool with 
the example generation.npy provided by the authors (MSE / correlation / visual pattern).

TODO: Then for Fukuoka, generate a new OD matrix using the same pipeline and discuss SDG relevance etc.
"""

data = np.load("assets/example_data/CommutingOD/GB_Liverpool/generation.npy", allow_pickle=True)
print(type(data), getattr(data, "shape", None))
# If it's an object array:
# print(data.item().keys())

print(data)

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
