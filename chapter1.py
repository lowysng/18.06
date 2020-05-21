import numpy as np

"""
vector_add          : add two vectors
vector_scale        : multiply vector by scalar
lin_comb            : compute the linear combination of a list of vectors
dot_product         : compute the dot product of two vectors
length              : compute the length of a vector 
unit_vector         : get the unit vector in the same direction as the vector provided
angle_btwn_vector   : apply cosine formula to get the angle between two vectors
matrix_vect_mult    : compute the matrix-vector multiplication of a matrix and a vector (Ax)

Example:
>>> A = np.array([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
>>> x = np.array([3, 2, 1])
>>> matrix_vect_mult(np.transpose(A), x)
[6, 9, 10]
"""

vector_add = lambda v, w: v + w
vector_scale = lambda v, c: c * v
lin_comb = lambda vects, scalars: np.add.reduce(vects*scalars[:, np.newaxis], axis=0)
dot_product = lambda v, w: np.sum(v * w)
length = lambda v: dot_product(v, v) ** 0.5
unit_vector = lambda v: v / length(v)
angle_btwn_vector = lambda v, w: np.arccos((dot_product(v, w)/(length(v) * length(w))))
matrix_vect_mult = lambda A, x: lin_comb(A, x)