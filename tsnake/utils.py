import numpy as np
from scipy.ndimage import gaussian_filter

"""
Module containing utilities and base classes to be shared by Snake and Grid
"""


def img_inflation_force(image: np.array, threshold: int) -> np.array:
    """
    Compute the inflationary force F(I(x, y)), equation (5) from the paper, for
    every integer-index (x, y) of the image.
    Args:
    ========================
    (np.array) image:
    * A np.array of shape (n, m) containing the grayscale image.

    (int) threshold:
    * Threshold value, intensities above this result in 1, else -1, from equation (5)
    ========================
    Returns:
    ========================
    (np.array) inflation forces (+1 or -1):
    * (n, m) array of of intensities (values of 0 to 255)
    ========================
    """
    image_intensity = np.copy(image)
    mask = image_intensity < threshold
    image_intensity[mask] = -1
    image_intensity[~mask] = 1
    return image_intensity


def img_force(image: np.array, sigma: float, c: float, p: float) -> np.array:
    """
    Computes an array of force values for the given image.
    Equation (7) in the paper.
    Args:
    ============================================
    (float) sigma:
    * The hyperparameter sigma from Equation (A.4).

    (float) c:
    * The hyperparameter c from Equation (A.4).

    (float) p:
    * The hyperparameter p from Equation (7).
    ============================================
    Returns:
    ============================================
    A np.array of shape (n, m, 2) containing the computed values of
    Equation (7) at each pixel in the image.
    ============================================
    """
    # Apply Gaussian smoothing
    smoothed = gaussian_filter(image, sigma=sigma)

    # Take the gradient
    x_grad, y_grad = np.gradient(smoothed)
    # Compute pixel-wise magnitudes
    mags = np.sqrt(np.square(x_grad) + np.square(y_grad))
    # Multiply by -c
    mags = mags * c

    x_grad, y_grad = np.gradient(mags)
    out = np.zeros((x_grad.shape[0], x_grad.shape[1], 2))
    out[:, :, 0] = x_grad
    out[:, :, 1] = y_grad

    # Scale the potential by p
    image_force = p * out
    return image_force


def dist(a: np.array, b: np.array) -> float:
    """
    Return the distance between a and b
    """
    return np.sqrt(np.sum(np.power(a-b, 2)))


def seg_intersect(a1, a2, b1, b2, decimal=3):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    Args: all are expected as (1,2) numpy arrays
    * a1: [x, y] a point on the first line
    * a2: [x, y] another point on the first line
    * b1: [x, y] a point on the second line
    * b2: [x, y] another point on the second line
    * decimal: int (optional): Number of decimals to round to, default is 3
    return:
    * (1,2) np array denoting [x, y] coordinates of intersection
    """
    s = np.vstack([a1, a2, b1, b2])     # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    result = None
    if z == 0:                          # lines are parallel
        result = np.array([float('inf'), float('inf')]).reshape(1, 2)
    else:
        result = np.array([x/z, y/z]).reshape(1, 2)
    return np.around(result, decimal)


class UtilPoint(object):
    """
    Represents a point on with x and y components
    """
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def update(self, x, y):
        self._x = x
        self._y = y

    @property
    def position(self):
        return np.array([self._x, self._y]).reshape(1, 2)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __str__(self):
        return '({}, {})'.format(self._x, self._y)

    def __repr__(self):
        return self.__str__()  # + ':' + str(self.__hash__()) # NOTE: For debugging, can add hash

    def __hash__(self):
        return hash((self._x, self._y))

    def __eq__(self, other):
        return self._x == other._x and self._y == other._y

    def __lt__(self, other):
        if self._x != other._x:
            return self._x < other._x
        else:
            return self._y < other._y

    def __sub__(self, other):
        return np.array([self._x - other._x, self._y - other._y], dtype=np.float32)

    def __add__(self, other):
        return np.array([self._x + other._x, self._y + other._y], dtype=np.float32)


class UtilEdge(object):
    """
    Represents one of the sides / cell-edges in the grid.
    """
    def __init__(self, point1: UtilPoint, point2: UtilPoint) -> None:
        """
        Represents one grid cell edge (one of three components of a TriangeCell).
        Args:
        ==========================================
        * point1: Point(), for first (origin) point of the line segment
        * point2: Point(), the terminal point of the line segment
        ==========================================
        """
        # pts = sorted([point1, point2])
        assert point1 != point2, 'Cannot initialize edge between the same point {}'.format(point1)
        self._point1 = point1  # todo maybe argchecks
        self._point2 = point2

    def get_perpendicular(self) -> np.array:
        """
        Return (2,) np array representing the perpendicular slope to this edge
        """
        x, y = self._point2 - self._point1
        return np.array([-y, x], dtype=np.float32)

    @property
    def endpoints(self):
        return np.array([self._point1, self._point2])

    def __str__(self):
        return '<{}, {}>'.format(str(self._point1), str(self._point2))

    def __repr__(self):
        return self.__str__() # + ':' + str(self.__hash__())

    def __hash__(self):
        return hash(tuple(sorted([self._point1, self._point2])))

    def __eq__(self, other):
        pts1 = sorted(self.endpoints)
        pts2 = sorted(other.endpoints)
        return np.all(pts1 == pts2)
