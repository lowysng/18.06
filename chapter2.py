import numpy as np
import chapter1 as c1 

def gaussian_elimination(augmented_matrix_, inplace=False):
    """Perform Gaussian elimination (in place) on input [A b], 
    where A is the coefficient matrix of a set of linear 
    equations and b is the RHS of the set of linear equations.
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
                    raise Exception('A is singular')
                temp_pivot_row = augmented_matrix[pivot_index + next_index]
                next_pivot_entry = temp_pivot_row[pivot_index]

            # exchange the rows
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

A = np.array([[2, 4, -2], [4, 9, -3], [-2, -3, 7]])
b = np.array([[2, 8, 10]])
augmented_matrix = np.transpose(np.vstack((A, b)))
gaussian_elimination(augmented_matrix, inplace=True)
print(augmented_matrix)