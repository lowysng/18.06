import numpy as np
import chapter1 as c1 

def gaussian_elimination(augmented_matrix_, inplace=False):
    """Perform Gaussian elimination on input [A b], 
    where A is the coefficient matrix representing a
    linear system and b is the RHS of the set of linear equations.

    Example: Solve Ax = b, where

            | 3  0  0 | 3 |
    [A b] = | 6  2  0 | 8 |
            | 9 -2  1 | 9 |

    >>> A = np.array([[3, 6, 9], [0, 2, -2], [0, 0, 1]])
    >>> b = np.array([[3, 8, 9]])
    >>> augmented_matrix = np.transpose(np.vstack((A, b)))
    >>> augmented_matrix
    [[ 3  0  0  3]
    [ 6  2  0  8]
    [ 9 -2  1  9]]
    >>> gaussian_elimination(augmented_matrix, inplace=True)
    >>> x = back_sub(augmented_matrix) # see back_sub() below
    >>> x
    [1. 1. 2.]
    >>> c1.lin_comb(A, x)
    [3. 8. 9.]  # our original b
    """
    if inplace:
        augmented_matrix = augmented_matrix_
    else:
        augmented_matrix = augmented_matrix_.copy()

    m = augmented_matrix.shape[0]       # number of rows of matrix A
    n = augmented_matrix.shape[1] - 1   # number of columns of matrix A
    pivot_index = 0

    while pivot_index < m:
        pivot_row = augmented_matrix[pivot_index]
        pivot_entry = pivot_row[pivot_index]
        if pivot_entry == 0:            
            # find row where pivot is not zero
            next_index = 0
            next_pivot_entry = 0
            while next_pivot_entry == 0:
                next_index += 1
                if pivot_index + next_index == m:
                    return augmented_matrix
                temp_pivot_row = augmented_matrix[pivot_index + next_index]
                next_pivot_entry = temp_pivot_row[pivot_index]
            # exchange rows
            temp = augmented_matrix[pivot_index].copy() 
            augmented_matrix[pivot_index] = temp_pivot_row
            augmented_matrix[pivot_index + next_index] = temp
            pivot_row = augmented_matrix[pivot_index]
            pivot_entry = pivot_row[pivot_index]
        # eliminate entries below the pivot
        elimination_index = pivot_index + 1
        while elimination_index < m:
            elimination_row = augmented_matrix[elimination_index]
            elimination_entry = elimination_row[pivot_index]
            multiplier = elimination_entry/pivot_entry
            augmented_matrix[elimination_index] = elimination_row - (pivot_row * multiplier)    
            elimination_index += 1
        pivot_index += 1
    return augmented_matrix

def back_sub(augmented_matrix):
    """Perform backsubstituion on input [U d],
    where U is the upper triangular matrix representing
    a linear system and d is the RHS of the set of linear
    equations. Too see input format, refer to gaussian_elimination().
    """
    m = augmented_matrix.shape[0]
    n = augmented_matrix.shape[1] - 1
    for i in range(m):
        assert augmented_matrix[i, i] != 0, "No solution/too many solutions: matrix U is singular"
        for j in range(i+1, m):
            assert augmented_matrix[j, i] == 0, "Matrix U must be in upper triangular form"
    x = np.zeros(n)
    augmented_matrix = np.flip(augmented_matrix)
    for i in range(m):
        current_row = augmented_matrix[i]
        d = current_row[0]
        U = current_row[1:]
        difference = d - sum((U*x)[:i])
        x[i] = difference / U[i]
    return np.flip(x)

def matrix_matrix_mult(A, B):
    return np.array([c1.lin_comb(A, col) for col in B])