import numpy as np

# Definir la matriz A y el vector b
A = np.array([[6, 2, 0],
              [-1, 0, 1],
              [8, 5, 0]])
b = np.array([8, 2, 13])

# Tolerancia y número máximo de iteraciones
tolerance = 1e-10
max_iterations = 1000

# Valores iniciales (suponer x1 = x2 = x3 = 0 al inicio)
x = np.zeros_like(b)

# Función de Jacobi
def jacobi(A, b, x, tolerance, max_iterations):
    D = np.diag(A)  # Diagonal de A
    R = A - np.diagflat(D)  # El resto de A (sin la diagonal)
    
    for i in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new, i
        x = x_new
    return x, max_iterations

sol, iterations = jacobi(A, b, x, tolerance, max_iterations)
print(f"Solución: {sol} en {iterations} iteraciones")
