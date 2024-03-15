import newton_factal
import numpy as np

def f(z):
    return z**3 - 1

def f_prime(z):
    return 3 * z**2

roots = np.array([1, -0.5 + 1j * np.sqrt(3) / 2, -0.5 - 1j * np.sqrt(3) / 2])
newton_factal.plot_newton_fractal(f, f_prime, roots, -10, 10, -10, 10, 1000, 1000, 1e-5, "./newton_fractal.png")