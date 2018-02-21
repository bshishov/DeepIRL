import numpy as np


def gaussian_2d(width: int, height: int, x: float, y: float, sigma: float = 0.05):
    x_axis, y_axis = np.meshgrid(
        np.linspace(-x, -x + 1.0, width),
        np.linspace(-y, -y + 1.0, height))
    d = np.sqrt(x_axis * x_axis + y_axis * y_axis)
    g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    return g


def append_2d_array_shifted(a: np.ndarray, b: np.ndarray, shift_b_x: int, shift_b_y: int):
    ha, wa = a.shape
    hb, wb = b.shape

    left_a = max(shift_b_x, 0)
    right_a = min(shift_b_x + wb, wa)
    top_a = max(shift_b_y, 0)
    bottom_a = min(shift_b_y + hb, ha)

    left_b = max(-shift_b_x, 0)
    right_b = min(wa - shift_b_x, wb)
    top_b = max(-shift_b_y, 0)
    bottom_b = min(ha - shift_b_y, hb)

    a[top_a:bottom_a, left_a:right_a] = a[top_a:bottom_a, left_a:right_a] + b[top_b:bottom_b, left_b:right_b]
    return a


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.random.random((100, 100))
    y = gaussian_2d(100, 100, 0.5, 0.5, 0.1)
    z = append_2d_array_shifted(x, y, 40, -40)
    plt.imshow(z)
    plt.show()



