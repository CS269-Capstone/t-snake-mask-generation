"""
Module containing implementations of the ACID technique.
"""

import numpy as np
import snake as snake
import cv2


class Point(object):
    """
    Represents a point on a grid with x and y components
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.edges = list()
    # TODO@allen: 
    # each point keeps track of edges and maintains set of all points
    # associated with the edges that start or terminate on that point
    # when adding a new edge between points, ensure that the pair of points doesn't
    # already have that edge, if it does, assign the one that already exists

    @property
    def position(self):
        return np.array([self.x, self.y]).reshape(1, 2)

    def __lt__(self, other):
        if self.x != other.x:
            return self.x < other.x
        else:
            return self.y < other.y


class GridCellEdge(object):
    """
    Represents one of the sides / cell-edges in the grid. 
    """

    def __init__(self, point1, point2):
        """
        Represents one grid cell edge (one of three components of a TriangeCell).

        Args:
        ==========================================
        (2-tuple) point1, point2:
        * The (x, y) points that form this line segment.
        * The top-left corner of the image-rectangle should be (0, 0).
          (to be compatible with numpy indexing) (can change this if inconvenient)
        ==========================================
        """
        self._point1 = point1  # todo maybe argchecks
        self._point2 = point2

        # TODO: implement this data structure
        self.intersections = dict()

    @property
    def endpoints(self):
        return {self._point1, self._point2}

    def find_intersection_point_with_element(self, element):
        """
        If this grid cell edge intersects with the given element:
            - Return the point of intersection (2-tuple of the coordinates).
        Else:
            - Return None
        """
        raise NotImplementedError


class Grid(object):
    """
    Class representing the entire cell grid (of triangles) for an image.
      - image meaning the blank space the T-snake is segmenting / infilling
      - assumes that each triangle-cell is a right triangle
        (for the Coxeter-Freudenthal triangulation) (see Fig 2 in the paper)

      - assumes (for now) that the space we're segmenting / infilling is rectangular


    In the paper, Demetri mentions the 'Freudenthal triangulation' for 
    implementing the cell-grid:
     https://www.cs.bgu.ac.il/~projects/projects/carmelie/html/triang/fred_T.htm

    Args:
    ==========================================
    (np.array) image:
    * (n by m) matrix representing the color image.

    (float) scale:
    * float between 0 and 1 representing the number of pixels per cell, i.e. 1=1 vertex/pixel, .5 = 2 vertex per pixel, so on

    ==========================================
    """

    def __init__(self, image, scale=1.0):
        """
        @allen: Should we pass a snake to the board? should the board own the snake?
        TODO: implement Freudenthal triangulation
        https://www.cs.bgu.ac.il/~projects/projects/carmelie/html/triang/fred_T.htm
        """
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # height * width * color channels

        # Raw image
        self.image = image
        self.m, self.n, self.d = image.shape

        # Image matrix after force and intensity function
        self.image_force = None
        self.image_intensity = None

        # Simplex Grid Vars

        if scale >= 1:
            s = "Scale > 1 must be an integer multiple of image size."
            assert self.m % scale == 0, s
            assert self.n % scale == 0, s
        elif scale > 0:
            inv = 1/float(scale)
            assert inv.is_integer(), "If scale < 1, 1/scale must be an integer"
        else:
            assert False, "Scale must be > 0."
        self.scale = scale
        self.grid = None

        # Hash set containing [Point(bottom left corner)]:all edges in pair of simplicies
        self.edges = dict()
        self.snakes = list()  # All the snakes on this grid
        # print("Grid initialized with:\n\theight: {}\n\twidth: {}\n\tdepth: {}".format(self.m, self.n, self.d))

    def appendToMap(self, map, key, item):
        """
        utility function to append item to map[key] if key in map
        """
        #TODO@allen: cleanup this function and it's signature
        if key in map:
            map[key].append(item)
        else:
            map[key] = list(item)

    def gen_simplex_grid(self):
        """
        Private method to generate simplex grid and edge map over image at given scale
        self.grid = np array of size (n/scale) * m/scale

        * Verticies are on if positive, off if negative, and contain 
            bilinearly interpolated greyscale values according to surrouding pixels

        * vertex position indicated by its x and y indicies
        """
        # NOTE: See todo in point class, this method is very much still under construction
        if self.scale <= 1:
            m_steps = int(self.m / self.scale)
            n_steps = int(self.n / self.scale)
            self.grid = np.zeros((m_steps, n_steps))
            for i in range(m_steps):
                for j in range(n_steps):
                    p2 = Point(m_steps*self.scale, n_steps*self.scale)
                    self.grid[i, j] = p2
                    if j > 0:
                        p1 = self.grid[i, j-1]
                        edge = GridCellEdge(p1, p2)
                        self.appendToMap(self.edges, p1, edge)
                    if i > 0:
                        p1 = self.grid[i-1, j]
                        edge = GridCellEdge(p1, p2)
                        self.appendToMap(self.edges, p1, edge)
                    if i > 0 and j > 0:
                        p1 = self.grid[i-1, j-1]
                        p_lower_left = self.grid[i-1, j]
                        p_upper_right = self.grid[i, j-1]
                        self.appendToMap(self.edges, p1, GridCellEdge(p_lower_left, p2))
                        self.appendToMap(self.edges, p1, GridCellEdge(p_upper_right, p2))
                        self.appendToMap(self.edges, p1, GridCellEdge(p_lower_left, p_upper_right))


        raise NotImplementedError

    def get_image_force(self, threshold):
        """
        Compute's force of self.image

        Args:
        ========================
        (int) threshold:
        * integer threshold, pixels with intensities above this value will be set to 1, else 0
        ========================
        Return:
        ========================
        (np.array) force: 
        * (self.image.shape[0] by self.image.shape[1]) boolean array of 0 and 1
        ========================
        """
        if self.image_force is None:
            intensity = self.get_image_intensity()
            self.image_force = np.zeros(intensity.shape) - 1
            self.image_force[intensity >= threshold] = 1

        return self.image_force

    def get_image_intensity(self):
        """
        Compute's intensity of self.image

        Args:
        ========================
        None
        ========================
        Return:
        ========================
        (np.array) intensities: 
        * (self.image.shape[0] by self.image.shape[1]) array of of intensities (values of 0 to 255)
        ========================
        """
        if self.image_intensity is None:
            self.image_intensity = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image_intensity

    def add_snake(self, new_snake):
        """
        Add a new snake to the grid
        """
        assert isinstance(new_snake, snake.Snake)
        self.snakes.append(new_snake)

    def compute_intersections(self, snake):
        """
        Compute intersections between the grid and the snake in question
        """
        # Get snake nodes, see which grid verticies they're closest to,
        # grab all the edges between the verticies they're closest to,
        # compute intersections
        # enqueue grid vertex inside the snake


### TODOS ###
# 1. snake updates - Joe
# 2. gan mask -> snake -> gan mask - Cole
# 3. algo phase 1: grid intersections - Allen
# 4. algo phase 2: turning nodes on / off, remove inactive points - Eric


if __name__ == '__main__':
    # Import testing
    snake = snake.Node(1, 1)
    edge = GridCellEdge(1, 1)

    # Manual Testing import
    img = cv2.imread("plane.png")
    grid = Grid(img, 1)
    grey = grid.get_image_intensity()
    force = grid.get_image_force(250)
    # print(grey, np.max(grey), grey.shape)
    print(force, np.max(force), force.shape)
    cv2.imshow("image", img)
    cv2.imshow("grey_image", grey)
    cv2.imshow("force_image", force)
    key = cv2.waitKey(0)

    pts = [[Point(1, 1), Point(1, 2)],
           [Point(1, 3), Point(1, 4)]]

    pts = np.array(pts)
    print(pts)
