import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def get_newton_factal_matrix(f, f_prime, roots, x_min, x_max, y_min, y_max, x_res, y_res, delta):
    x = np.linspace(x_min, x_max, x_res)
    y = np.linspace(y_min, y_max, y_res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # don't iterate over a fixed number of iterations, but until the difference between the current and the previous value is smaller than delta
    while True:
        Z_next = Z - f(Z) / f_prime(Z)
        if np.all(np.abs(Z_next - Z) < delta):
            break
        Z = Z_next
    
    # assign a interger value that is the inidex of the root that the point converges to
    Z_index = np.zeros(Z.shape, dtype=int)
    #assign each point ot the closest root
    diff = np.zeros(roots.shape + Z.shape)
    for i in range(roots.size):
        diff[i] = np.abs(Z - roots[i])
    Z_index = np.argmin(diff, axis=0)

    

    return Z_index

def plot_newton_fractal(f, f_prime, roots, x_min, x_max, y_min, y_max, x_res, y_res, delta, output_path):
    # Assuming get_newton_fractal_matrix is implemented correctly
    Z_index = get_newton_factal_matrix(f, f_prime, roots, x_min, x_max, y_min, y_max, x_res, y_res, delta)
    
    # Convert the integer index to an RGB color using HSV space for better color differentiation
    hsv = np.zeros(Z_index.shape + (3,))
    unique_indices = np.unique(Z_index)
    max_index = len(unique_indices)
    
    # Map each unique index to a unique hue value
    for index in unique_indices:
        hsv[Z_index == index, 0] = 360 * index / max_index
        hsv[Z_index == index, 1] = 1
        hsv[Z_index == index, 2] = Z_index[Z_index == index] / np.max(Z_index) # Optional: Vary brightness
    
    # Convert HSV to RGB since imshow expects RGB values
    rgb = hsv_to_rgb(hsv)
    
    # Make the image
    plt.imshow(rgb, extent=(x_min, x_max, y_min, y_max), origin='lower')
    
    # Also plot the roots
    plt.scatter(roots.real, roots.imag, color='white', s=50, edgecolor='black')  # White to stand out, with black border
    
    # Save the plot to a file
    plt.savefig(output_path)


