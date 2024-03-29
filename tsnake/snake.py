"""
Module containing implementation of parts of the paper, up to and including Section 3.2
"""

import numpy as np
from scipy.linalg import solve_triangular
from .utils import UtilPoint as uPoint
from .utils import UtilEdge as uEdge
from .utils import dist, seg_intersect

# Implementation Notes: https://www.crisluengo.net/archives/217#more-217


class Node(uPoint):
    """
    A class representing a single node in a T-snake.
    """
    def __init__(self, x, y):
        super().__init__(x, y)
        self._normal = None

    def set_normal(self, norm):
        assert isinstance(
            norm, np.ndarray), 'Norm was type {}'.format(type(norm))
        assert norm.shape == (
            2,), 'Norm should be np array of (2,), got {}'.format(norm.shape)
        assert norm is not None
        self._normal = norm

    @property
    def normal(self):
        return self._normal


class Element(uEdge):
    """
    Class representing an element / edge between two nodes in the T-snake.
    The T-Snake class instantiates the Elements automatically in its constructor,
     so directly calling this constructor elsewhere (probably) shouldn't be necessary.
     (NOTE: can change if inconvenient)
    """
    def __init__(self, node1, node2):
        assert isinstance(node1, Node)
        assert isinstance(node2, Node)
        super().__init__(node1, node2)
        self._node1 = node1
        self._node2 = node2
        self._normal = None

    @property
    def nodes(self):
        return np.array([self._node1, self._node2])

    @property
    def normal(self):
        return self._normal

    def set_normal(self, norm):
        assert isinstance(
            norm, np.ndarray), 'Norm was type {}'.format(type(norm))
        assert norm.shape == (
            2,), 'Norm should be np array of (2,), got {}'.format(norm.shape)
        assert norm is not None
        self._normal = norm

    def intersects_grid_cell(self, grid_cell):
        # TODO: clean this up
        raise NotImplementedError


class TSnake(object):
    """
    A class representing a *single* T-snake at a given instance in time.
    If the initial T-snake splits into two, then there should be *two*
      instances of T-snakes present in the algorithm.
      (NOTE: can change this if inconvenient)
    The merging two T-snakes in the algorithm should be done with:
      TSnake.merge(snake1, snake2)
    In this class,
        each element/edge       self.elements[i]
            corresponds to:
        node pair               (self.nodes[i], self.nodes[i+1])
    Args:
    ===========================================
    (list) nodes:
    * A list of Node instances, in order i=0, 1, 2, ..., N-1
    ===========================================
    TODO: calculate intensity normal thingy
    TODO: comments for a, b, gamma
    """
    def __init__(self, nodes, force, intensity, a, b, q, gamma, dt):
        for n in nodes:
            assert isinstance(n, Node)
        self.nodes = list(nodes)

        # force: (n, m, 2) array of Eq (7) at each pixel (x, y)
        self.force = force

        # NOTE: this being called 'intensity' is confusing - it's
        #       actually the inflationary force
        # intensity: (n, m) array of Eq (5) at each pixel (x, y)
        self.intensity = intensity

        # Deformation parameters
        self.a = a
        self.b = b
        self.gamma = gamma
        self.dt = dt
        self.q = q
        self.params = [self.a ,self.b ,self.gamma ,self.dt ,self.q]

        self._elements = []

        # Connect each node[i] --> node[i+1]
        for i in range(len(nodes)-1):
            self._elements.append(Element(self.nodes[i], self.nodes[i+1]))

        # Connect node[N-1] --> node[0]
        self._elements.append(Element(self.nodes[-1], self.nodes[0]))
        self._compute_normals()

    def _remove_empty_elements(self):
        new_elements = []
        for element in self._elements:
            p1, p2 = element.endpoints
            if p1 != p2:
                new_elements.append(element)
        del self._elements
        self._elements = new_elements

    def _compute_normals(self):
        """
        Compute normals for each element and node, the computation is O(n)
        in the number of edges.
        Args:
        =====================================================
        None, uses the initialized snake elements and nodes, but
        expects elements to be initialized counter clockwise for
        any closed contour
        =====================================================
        Returns:
        =====================================================
        None, stores the normal to each element and node as a
        numpy array of dimension (2,), i.e. [x y]
        =====================================================
        """
        self._remove_empty_elements()

        for i in range(len(self._elements)):
            first_element = self._elements[i]
            p1, p2 = first_element.endpoints

            # Perpendicular of current element, normalized
            norm = first_element.get_perpendicular().reshape(2,)
            norm /= np.sum(np.abs(norm))

            first_element.set_normal(norm)
            # Now we find the normals for each node, which
            # are the average of the normals of the two adjacent elements
            if i > 0:
                prev = self._elements[i-1]
                current = self._elements[i]
                self.nodes[i].set_normal(0.5 * (prev.normal + current.normal))

        # Do the normal of the first node now that the final element
        # has a normal vector too
        prev = self._elements[-1]
        current = self._elements[0]
        self.nodes[0].set_normal(0.5 * (prev.normal + current.normal))

    @property
    def elements(self):
        return np.array(self._elements)

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def node_locations(self):
        """
        Returns an (N, 2) matrix containing the current node locations for this T-snake.
        ( In the paper: $\bm{x}(t)$ )
        """
        N = self.num_nodes
        locs = [node.position for node in self.nodes]
        return np.array(locs).reshape(N, 2)

    def compute_matrix(self):
        """
        Computes N x N matrix P = (I - (dt/gamma)*A)
        Returns the Cholesky Decomposition of P, L : N x N lower triangle matrix
        Follows from: https://www.crisluengo.net/archives/217
        """
        N = self.num_nodes
        d0 = (self.dt/self.gamma)*(2*self.a + 6*self.b) + 1
        d1 = (self.dt/self.gamma)*(-self.a - 4*self.b)
        d2 = (self.dt/self.gamma)*(self.b)

        # construct diagonal 0
        P = np.diag(np.repeat(d0, N))
        # add diagonal 1 and diagonal -(N-1), which is bottom left corner
        P = P + np.diag(np.repeat(d1, N-1), k=1) + \
            np.diag(np.array([d1]), k=-N+1)
        # add diagonal -1 and diagonal (N-1), which is upper right corner
        P = P + np.diag(np.repeat(d1, N-1), k=-1) + \
            np.diag(np.array([d1]), k=N-1)
        # add diagonal 2 and diagonal -(N-2)
        P = P + np.diag(np.repeat(d2, N-2), k=2) + \
            np.diag(np.array([d2, d2]), k=-N+2)
        # add diagonal -2 and diagonal (N-2)
        P = P + np.diag(np.repeat(d2, N-2), k=-2) + \
            np.diag(np.array([d2, d2]), k=N-2)

        # calculate lower trangle matrix of Cholesky decomposition
        L = np.linalg.cholesky(P)

        return L

    def bilinear_interpolate(self, im, x, y):
        """
        Computes bilinearly interpolated values for points (x,y) from image im.
        Follows (and modified) from Alex Flint's code:
        https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
        Args:
        ===========================================
        (2D numpy array) im:
        * 2D numpy array containg the image values for interpolation.

        (numpy array) x:
        * array of x-coordinates to be interpolated.

        (numpy array) y:
        * array of y-coordinates to be interpolated.
        ===========================================
        """
        # find the top left point to interpolate from
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)

        # make sure the coordinates are within the image
        # @cole: will a correct implementation ever call this function on coords
        #        outside of the image? shouldn't we throw an error if this happens?
        x0 = np.clip(x0, 0, im.shape[0]-1)
        y0 = np.clip(y0, 0, im.shape[1]-1)

        # if the top left point is on the edge of the image,
        # subtract 1 from the coordinate
        x0[x0 == im.shape[0]-1] -= 1
        y0[y0 == im.shape[1]-1] -= 1

        # find the bottom right point
        x1 = x0 + 1
        y1 = y0 + 1

        Ia = im[x0, y0]
        Ib = im[x0, y1]
        Ic = im[x1, y0]
        Id = im[x1, y1]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        denom = (x1-x0)*(y1-y0)

        if np.any(denom == 0):
            raise ValueError('Division by zero in bilinear interpolation')

        return (wa*Ia + wb*Ib + wc*Ic + wd*Id)/denom

    def m_step(self, M):
        """
        Runs the M iterations of the snake evolution on the N nodes of the snake.
        Args:
        ===========================================
        (int) M:
        * Non-negative int representing the number of iteration steps.
        ===========================================
        """
        L = self.compute_matrix()

        X = np.zeros((self.num_nodes, 1))
        Y = np.zeros((self.num_nodes, 1))
        norms = np.zeros((self.num_nodes, 2))

        for i in range(self.num_nodes):
            pos = self.nodes[i].position
            norms[i] = self.nodes[i].normal

            X[i][0] = pos[0][0]
            Y[i][0] = pos[0][1]

        for i in range(M):
            # TODO: Update assumptions below, now that force and normals
            # have been calculated

            # self.force is the external potential force - Eq (7)
            fx = self.bilinear_interpolate(self.force[:, :, 0], X, Y)
            fy = self.bilinear_interpolate(self.force[:, :, 1], X, Y)

            # self.intensity is the inflation force - Eq (5)
            pxy = self.bilinear_interpolate(self.intensity, X, Y)
            # Get component of intensity on x and y directions
            px = pxy * self.q * (norms[:, 0]).reshape(-1, 1)
            py = pxy * self.q * (norms[:, 1]).reshape(-1, 1)

            # Update nodes
            temp = solve_triangular(
                L, X + (self.dt/self.gamma)*(fx + px), lower=True
            )
            X = solve_triangular(np.transpose(L), temp, lower=False)

            temp = solve_triangular(
                L, Y + (self.dt/self.gamma)*(fy + py), lower=True
            )
            Y = solve_triangular(np.transpose(L), temp, lower=False)

            # make sure X and Y are within image
            X = np.clip(X, 0, self.force.shape[0]-1)
            Y = np.clip(Y, 0, self.force.shape[1]-1)

            self._compute_normals()

        # save new nodes
        for i in range(self.num_nodes):
            self.nodes[i].update(X[i], Y[i])

    @classmethod
    def merge(cls, snake1, snake2):
        """
        Merge two (previously-split) T-snakes together.
        """
        raise NotImplementedError

    def update_snake_nodes(self, new_nodes):
        """
        Updates snake with new nodes defined in np array of length n
        new_nodes of Node() objects
        """
        raise NotImplementedError


if __name__ == '__main__':
    pass
