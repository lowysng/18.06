import sys, re
import numpy as np
import chapter1 as c1
import chapter2 as c2

if len(sys.argv) < 2:
    print('Usage: python app.py linear-sys [Matrix A] [Vector b]')
    print('To represent A, write each columns of A, separated by semicolons')
    print('Example: write matrix A = \n|1 1|\n|1 1|\nas 1,1;1,1')

elif sys.argv[1] == 'linear-sys':    
    raw_matrix = sys.argv[2]
    raw_b = sys.argv[3]
    rows = raw_matrix.split(';')
    A = np.array([[int(x) for x in row.split(',')] for row in rows])
    b = np.array([int(x) for x in raw_b.split(',')])
    augmented_matrix = np.transpose(np.vstack((A, b)))
    U_d = c2.gaussian_elimination(augmented_matrix)
    x = c2.back_sub(U_d)
    print('--------------------')
    print('Ax = b where [A b] =\n', augmented_matrix)
    print('--------------------')
    print('Solution x =\n', x)
    print('--------------------')
