import numpy as np
import cv2
from .snake import TSnake, Element, Node
from .utils import UtilPoint as uPoint
from .utils import UtilEdge as uEdge
from .utils import dist, seg_intersect, img_force, img_inflation_force
from typing import List
from scipy.spatial.distance import cdist
from collections import deque
TSnakes = List[TSnake]

"""
Module containing implementations of the ACID technique.
"""


class Point(uPoint):
    """
    Represents a point on a grid with x and y components
    """

    def __init__(self, x, y):
        super().__init__(x, y)
        self.adjacent_edges = dict()
        self._is_on = False # Initialize to on because our snake shrinks 
        self.sign = None

    @property
    def is_on(self):
        return self._is_on

    # Once a node is on, it stays on
    # def turn_off(self):
    #     self._is_on = False

    def turn_on(self):
        self._is_on = True

    def add_edge(self, edge):
        '''
        Add given edge to the dict of edges connected to this node
        args:
        * edge: GridCellEdge connected to this node
        return:
        * None: Stores edge in self.adjacent_edges dictionary
        '''
        self.adjacent_edges[edge] = edge


class GridCellEdge(uEdge):
    """
    Represents one of the sides / cell-edges in the grid.
    """

    def __init__(self, point1: Point, point2: Point) -> None:
        """
        Represents one grid cell edge (one of three components of a TriangeCell).
        Args:
        ==========================================
        * point1: Point(), for first (origin) point of the line segment
        * point2: Point(), the terminal point of the line segment
        ==========================================
        """
        super().__init__(point1, point2)
        self.intersections = list()

    def add_intersection(self, point: Point, element: Element) -> Point:
        """
        Store the intersection in the edge
        Args:
        -------------------------------
        * point: Point(), point object representing where the intersection occured
        * element: Element(), element that the grid intersected with
        -------------------------------
        Return:
        ==========================================
        * Point, the vertex in the ACID grid if if is inside the snake, 
        else None if it is not inside the snake
        ==========================================
        """
        # Get the minimum vertex of this grid cell edge
        max_vertex = max(self._point1, self._point2)

        # Get the direction from the intersect to the vertex
        plus_norm = point.position.reshape(-1) + element.normal
        minus_norm = point.position.reshape(-1) - element.normal
        
        plus_dist = dist(plus_norm, max_vertex.position.reshape(-1))
        minus_dist = dist(minus_norm, max_vertex.position.reshape(-1))
        
        is_outside = plus_dist < minus_dist
        if is_outside:
            point.sign = 1  # or -1? depending on the normal?
        else:
            point.sign = -1
        
        if not self.intersections:
            self.intersections.append(point)
        elif self.intersections[-1].sign == point.sign:
            self.intersections[-1] = point
        else:
            self.intersections = []
        
        return_val = None
        if is_outside:
            return_val = max_vertex
            max_vertex.turn_on()

        return return_val

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
        assert isinstance(
            image, np.ndarray), 'Image is of type: {}'.format(type(image))
        # assert len(image.shape) == 3  # height * width * color channels

        # Raw image
        self.image = image
        self.m, self.n = image.shape

        # Image matrix after force and intensity function
        self.image_force = None
        self.image_inflation_force = None

        # # Simplex Grid Vars
        # if scale >= 1:
        #     s = 'Scale > 1 must be an integer multiple of image size.'
        #     assert self.m % scale == 0, s
        #     assert self.n % scale == 0, s
        # elif scale > 0:
        #     inv = 1/float(scale)
        #     assert inv.is_integer(), 'If scale < 1, 1/scale must be an integer'
        # else:
        #     assert False, 'Scale must be > 0.'
        self.scale = scale
        self.grid = None

        # Hash map containing [Point(upper left corner)]:all edges in pair of simplicies
        self.point_edge_map = dict()
        self.edges = dict()  # set of all edges
        # print('Grid initialized with:\n\theight: {}\n\twidth: {}\n\tdepth: {}'.format(self.m, self.n, self.d))
        
        self.gen_simplex_grid()

    def _store_edge(self, p1: Point, p2: Point) -> None:
        """
        Store the edge between Points p1 and p2 in both p1 and p2,
        unless the edge already exists, then that edge is used.
        Args:
        ==========================================
        * Point: p1, p2: two points to store edge between
        ==========================================
        """
        edge = GridCellEdge(p1, p2)
        if edge in self.edges:
            edge = self.edges[edge]
        else:
            self.edges[edge] = edge
        p1.add_edge(edge)
        p2.add_edge(edge)

    def gen_simplex_grid(self):
        """
        Method to generate simplex grid and edge map over image at given scale
        self.grid = np array of size (n/scale) * m/scale
        * Vertices are on if positive, off if negative, and contain
            bilinearly interpolated greyscale values according to surrouding pixels
        * vertex position indicated by its x and y indicies
        """
        m_steps = None
        n_steps = None

        m_steps = int(self.m / self.scale) + 1
        n_steps = int(self.n / self.scale) + 1

        self.grid = np.empty((m_steps, n_steps), dtype=object)
        for i in range(m_steps):
            for j in range(n_steps):
                curr_pt = Point(i*self.scale, j*self.scale)
                self.grid[i, j] = curr_pt
                if j > 0:
                    p1 = self.grid[i, j-1]
                    self._store_edge(curr_pt, p1)  # horizontal edge

                if i > 0:
                    p1 = self.grid[i-1, j]
                    self._store_edge(curr_pt, p1)  # vertical edge
                    if j < n_steps - 1:
                        p2 = self.grid[i-1, j+1]
                        self._store_edge(curr_pt, p2)  # diagnoal edge
                        
        return self.grid

    def get_image_force(self, sigma, c, p):
        """
        Compute's force of self.image
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
        if self.image_force is None:
            self.image_force = img_force(self.image, sigma, c, p)
        return self.image_force

    def get_inflation_force(self, threshold):
        """
        Compute F(I(img)), equation (5) from the paper, for inflation force.
        Args:
        ========================
        (int) threshold:
        * Threshold value, intensities above this result in 1, else -1, from equation (5)
        ========================
        Returns:
        ========================
        (np.array) inflation forces (+1 or -1):
        * (self.image.shape[0] by self.image.shape[1]) array of of intensities (values of 0 to 255)
        ========================
        """
        if self.image_inflation_force is None:
            self.image_inflation_force = img_inflation_force(
                self.image, threshold)
        return self.image_inflation_force

    def get_closest_node(self, position: np.array) -> np.array:
        """
        Get the closest grid point to the coordinates
        of the snake node passed in position
        args:
        * np array (1,2) containing x and y position of node
        return:
        * np array (1,2) containing self.grid index of closest grid point
        """
        # Integer divide to closest node
        pos_frac = position - np.fix(position)
        pos_whole = position - pos_frac
        remainder = np.fmod(pos_frac, self.scale)
        idx = np.array((position-remainder)/self.scale, dtype=int)
        return idx

    def get_cell_edges(self, index):
        """
        get all edges bounded by the box with the index
        position as it's top-left corner
        Args:
        ==========================================
        * index: np array (1,2) of index of bounding box's top-left corner
        ==========================================
        Returns:
        ==========================================
        * edges: set() of all edges bounded by this box,
                 i.e., potential intersection points
        ==========================================
        """
        edges = set()
        xlim, ylim = self.grid.shape
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = index[0, 0]+dx, index[0, 1]+dy
                if x < 0 or x >= xlim or y < 0 or y >= ylim:
                    continue
                pt = self.grid[x, y]
                for key in pt.adjacent_edges:
                    edges.add(pt.adjacent_edges[key])
        return edges

    def _get_element_intersection(self, element: Element, edge: GridCellEdge) -> Point:
        """
        Get intersection between snake element and grid-cell-edge
        Args:
        * element: snake element, edge: GridCellEdge
        Return:
        * Point: intersection point, or None if no intersection
        """
        s1, s2 = element.nodes  # TODO: Add to snake, (1,2) np array of [dx, dy]
        e1, e2 = edge.endpoints

        # Find intersection candidate
        intersection = seg_intersect(
            s1.position, s2.position, e1.position, e2.position)

        # Check if the two lines are parallel
        if intersection[0, 0] == float('inf'):
            return None

        # Check if it's too far from the snake element endpoints to be valid
        ds1 = dist(intersection, s1.position)
        ds2 = dist(intersection, s2.position)
        d_snake = dist(s1.position, s2.position)
        if ds2 > d_snake or ds1 > d_snake:
            return None

        # Check if it's too far from the GridCellEdge endpoints to be valid
        de1 = dist(intersection, e1.position)
        de2 = dist(intersection, e2.position)
        d_edge = dist(e1.position, e2.position)
        if de1 > d_edge or de2 > d_edge:
            return None

        return Point(intersection[0, 0], intersection[0, 1])

    def _compute_intersection(self, snake: TSnake) -> ([GridCellEdge], deque([Point])):
        """
        Compute intersections between the grid and the snake in question
        Args:
        * snake: TSnake to compute intersections with
        Return:
        * [Point]: contains all found intersection points. 
        These points already have points resulting from multiple
        intersections on the same edge removed
        * deque([Point]): Queue of Points outside of the shrunken snake for
        processing in reparametrization phase 2
        """

        elements = snake.elements
        intersections = []
        grid_node_queue = deque()
        # All intersection points that have been found, mapped to edges they were found on
        intersect_set = dict()
        checked_edges = set()

        for element in elements:
            # Get all edges surrounding each node, and check each for intersections
            node1, _ = element.nodes
            index = self.get_closest_node(node1.position)
            edges_to_check = list(self.get_cell_edges(index))
            while edges_to_check:
                edge = edges_to_check.pop()

                intersect_pt = self._get_element_intersection(element, edge)
                if intersect_pt is not None and intersect_pt not in intersect_set:
                    # intersections.append(intersect_pt)
                    grid_node = edge.add_intersection(intersect_pt, element)
                    if grid_node is not None:
                        grid_node_queue.append(grid_node)
                    intersect_set[intersect_pt] = edge
                    checked_edges.add(edge)

                    # Add all potential new edges to check to the stack
                    # Duplicate intersections won't be added due to the above rule
                    index = self.get_closest_node(intersect_pt.position)
                    new_edges_to_check = list(self.get_cell_edges(index))
                    for new_edge in new_edges_to_check:
                        edges_to_check.append(new_edge)

                    # # NOTE: Code to debug intersection points, see known bug above
                    # if np.sum(intersect_pt.position - node1.position) == 0 or np.sum(intersect_pt.position - node2.position) == 0:
                    #     print("Following intersection goes through existing point:")
                    # print("Edge: {}, Node1({}, {}), Node2({}, {}), index: {}, intersect point: {}".format(
                    #     edge, node1.position[0, 0], node1.position[0,1], node2.position[0, 0], node2.position[0, 1],
                    #     index, intersect_pt
                    # ))

        for edge in checked_edges:
            for point in edge.intersections:
                intersections.append(point)
        return (intersections, grid_node_queue)

    def _sort_nodes(self, nodes):
        # Pre-process sort to get right-handed spiral for normal init
        # This is necessary to get normal vectors to initialize properly
        # nodes.sort(key=lambda x: (x[0] + x[1], x[0], x[1]), reverse=True)

        num_nodes = len(nodes)
        node_locs = np.array(nodes).reshape(num_nodes, 2)

        # Pull out arbitrary starting node
        loc = node_locs[0]
        node_locs = node_locs[1:]

        ordered_nodes = [loc]
        while len(ordered_nodes) < num_nodes:
            # Find the node which is closest to the last node we processed
            last = ordered_nodes[-1].reshape(1, 2)
            # dists = distance from each remaining node to current node
            dists = cdist(last, node_locs).reshape(-1, )

            closest = dists.argmin()
            ordered_nodes.append(node_locs[closest])
            node_locs = np.delete(node_locs, obj=closest, axis=0)

        # Source: https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
        temp = np.array(ordered_nodes).reshape(-1, 2)
        x = temp[:, 0].reshape(-1)
        y = temp[:, 1].reshape(-1)
        xs, ys = np.zeros(x.shape), np.zeros(y.shape)
        for i in range(x.shape[0]):
            xs[i] = x[i] - x[i-1]
            ys[i] = y[i] + y[i-1]

        orientation = np.sum(xs*ys)
        if orientation < 0:
            ordered_nodes = ordered_nodes[::-1]

        return ordered_nodes

    def get_snake_intersections(self, snakes: TSnakes) -> [[Point]]:
        """
        Compute intersections between all snakes on the grid and the grid.
        Args:
        * List of T-Snakes
        Return:
        * list(list(Point)) containing the intersection points for each snake
        """
        return [self._compute_intersection(snake)[0] for snake in snakes]

    def reparameterize_phase_one(self, snakes: [TSnake]) -> ([TSnake], deque([Point])):
        '''
        Takes a list of T-Snakes and returns a new list of T-Snakes (which have possibly been split/merged/etc).
        If no snakes are split/merged, then the snakes are presented in the same order as before, AND
        each snake's nodes are presented in the same order.
        
         Arguments:
        ===================================
        * snakes: List of TSnakes to be reparametrized
        ===================================
        Returns:
        ===================================
        * tuple([Tsnake], [deque([Point])]): 
        ** [Tsnake] is a list of the reparametrized TSnakes (same order if no splits or merges)
        ** [deque([Point])] is a list of queue of points in the ACID grid that will need to be processed in 
        reparametrization phase II
        ===================================
        '''
        new_snakes = []
        grid_node_queues = []
        for snake in snakes:
            intersections, grid_node_queue = self._compute_intersection(snake)

            new_nodes = [[i.position[0, 0], i.position[0, 1]]
                         for i in intersections]
            new_nodes = self._sort_nodes(new_nodes)
            new_nodes = [Node(x[0], x[1]) for x in new_nodes]
            a, b, gamma, dt, q = snake.params
            force, intensity = snake.force, snake.intensity
            new_snake = TSnake(nodes=new_nodes, force=force,
                               intensity=intensity, a=a, b=b, q=q, gamma=gamma, dt=dt)

            new_snakes.append(new_snake)
            grid_node_queues.append(grid_node_queue)

            # NOTE: Create new snake nodes in same order as intersections were computed.
            # Initialize snake in same order as intersections.
            # If the snake split, add the newly created snakes too.

        return (new_snakes, grid_node_queues)


    def reparameterize_phase_two(self, snakes: [TSnake], grid_node_queues: [deque([Point])]) -> [TSnake]:
        '''
        Takes a list of ACID grid verticies found in reparameterize_phase_one()
        and performs reparametrize phase two. It then returns a new list of T-Snakes (which have 
        possibly been split/merged/etc). If no snakes are split/merged, then the snakes are 
        presented in the same order as before, AND each snake's nodes are presented in the same order.
        
         Arguments:
        ===================================
        * grid_node_queues: list(deque([Point])) to be dequeued and processed, 1 for each
        snake that was passed to reparametrize_phase_1
        * snakes: List of the old Tsnakes
        ===================================
        Returns:
        ===================================
        * [Tsnake] is a list of the reparametrized TSnakes (same order if no splits or merges)
        ===================================
        '''
        # Nodes that have just been turned on, i.e.
        # new outside boundary of the snake
        for grid_node_queue in grid_node_queues:
            # intersection_nodes = list(grid_node_queue) 
            # on_nodes = []
            while grid_node_queue:
                node = grid_node_queue.popleft()
                if node.is_on:
                    for edge in node.adjacent_edges:
                        if len(edge.intersections) == 0:
                            for p in edge.endpoints:
                                if not p.is_on:
                                    p.turn_on()
                                    grid_node_queue.append(p)
                    # on_nodes.append(p)
            




if __name__ == '__main__':
    pass
