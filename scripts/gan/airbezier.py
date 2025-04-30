import numpy as np
from scipy.special import comb

# Function to generate a Bezier curve from a set of control points
def bezier(points, n_interp=100):
    points = points.copy()
    points[-1] = points[0]  # Close loop
    midpoint = (points[1] + points[-2]) / 2
    points[1] = midpoint
    points[-2] = midpoint

    def bernstein_poly(i, n, t):
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    n = len(points) - 1
    t = np.linspace(0.0, 1.0, n_interp)
    curve = np.zeros((n_interp, 2))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), points[i])
    return curve

# Function to normalize the airfoil points to a standard chord length of 1
# and to align the leading edge with the origin and the trailing edge along the x-axis
def normalise_airfoil(points):
    points = points.copy()
    le_idx = np.argmin(points[:, 0])
    te_idx = np.argmax(points[:, 0])
    le, te = points[le_idx], points[te_idx]
    translated = points - le
    chord_vec = te - le
    angle = -np.arctan2(chord_vec[1], chord_vec[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    rotated = translated @ rot.T
    return rotated / rotated[te_idx][0]
