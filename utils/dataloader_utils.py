import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import random

def interpolate_table(path="", matrix_width=81, matrix_height=41, relative_values="segment_pressure_value (MPa)"):
    data = pd.read_excel(path)

    valid_data = data.dropna(subset=[relative_values])
    coordinates = valid_data[["x (mm)", "y (mm)"]].values
    rel_values = valid_data[relative_values].values

    x_uniform = np.linspace(data["x (mm)"].min(), data["x (mm)"].max(), matrix_width)

    y_uniform = np.linspace(data["y (mm)"].min(), data["y (mm)"].max(), matrix_height)

    x_grid, y_grid = np.meshgrid(x_uniform, y_uniform)

    pressure_matrix_interpolated = griddata(coordinates, rel_values, (x_grid, y_grid), method="linear", fill_value=0)

    # pressure_matrix_interpolated[pressure_matrix_interpolated < 0] = 0

    return pd.DataFrame(pressure_matrix_interpolated, columns=x_uniform, index=y_uniform).to_numpy(dtype=np.float32)

def resize_matrix(matrix, new_width=41, new_height=81):
    old_height, old_width = matrix.shape

    old_x = np.linspace(0, 1, old_width)
    old_y = np.linspace(0, 1, old_height)
    new_x = np.linspace(0, 1, new_width)
    new_y = np.linspace(0, 1, new_height)

    old_x_grid, old_y_grid = np.meshgrid(old_x, old_y)
    new_x_grid, new_y_grid = np.meshgrid(new_x, new_y)

    old_points = np.vstack((old_x_grid.ravel(), old_y_grid.ravel())).T
    new_points = np.vstack((new_x_grid.ravel(), new_y_grid.ravel())).T

    resized_matrix = griddata(old_points, matrix.ravel(), new_points, method="linear", fill_value=0)

    return resized_matrix.reshape(new_height, new_width)

def random_flip(arrays):
    horizontal_flip = random.random() > 0.5
    vertical_flip = random.random() > 0.5

    if horizontal_flip:
        arrays = [np.fliplr(array) for array in arrays]

    if vertical_flip:
        arrays = [np.flipud(array) for array in arrays]

    return tuple(arrays)
