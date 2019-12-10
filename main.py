import cv2
import numpy as np
import matplotlib.pyplot as plt

import tsnake.initialize as init
from tsnake.snake import TSnake, Element, Node
from tsnake.grid import Grid, Point
from tsnake.utils import dist, seg_intersect

if __name__ == "__main__":
    msk_path = 'examples/places2/case1_mask.png'
    img_path = 'examples/places2/case1_raw.png'
    mask = init.load_mask(msk_path)
    image = init.load_grayscale_image(img_path)

    regions = init._find_disjoint_masked_regions(mask)

    # NOTE: Uncomment to visialize initial masked reigons
    # init.visualize_masked_regions(mask, regions)

    regions = init.compute_masked_regions(image, mask)
    tsnakes = []
    for region in regions:
        tsnake = region.initialize_tsnake(
            N=1000, p=1.0, c=1.0, sigma=1.0, a=1.0, b=1.0, gamma=1.0,
            dt=0.01
        )
        tsnakes.append(tsnake)
        # region.visualize() # NOTE: To show tsnakes on images, uncomment

    tsnakes.sort(key=lambda t: len(t.nodes))
    print("Length of T-Snakes initialized on image (sorted):\n{}".format(
        [len(t.nodes) for t in tsnakes]))

    image = init.load_grayscale_image(img_path)
    print(image.shape)
    grid = Grid(image=image, scale=4)

    # Update grid
    # NOTE: Uncomment for force, expensive calculation
    # force = grid.get_image_force(2,2,2)
    grid.gen_simplex_grid()
    print("Simplex Grid shape: {}".format(grid.grid.shape))

    # Add snakes to grid
    for s in tsnakes:
        grid.add_snake(s)
        break

    # Compute snake intersections with grid
    intersections = grid.get_snake_intersections()
    print(intersections)

    if False:  # Manual tests of grid
        # Import testing
        positions = [(0.9, 0.9), (1.1, 0.9), (1.1, 1.1), (0.9, 1.1)]
        nodes = [Node(p[0], p[1]) for p in positions]

        # NOTE: Manual Testing for image functions
        # Replace plane.png with any image locally in the folder
        img = cv2.imread("tsnake/plane.png")
        grid = Grid(img, 0.5)
        grey = grid.get_image_intensity()
        force = grid.get_image_force(250)

        snake = TSnake(nodes, force, grey, 1, 1, 1, 1)

        cv2.imshow("image", img)
        cv2.imshow("grey_image", grey)
        cv2.imshow("force_image", force)
        key = cv2.waitKey(0)

        pts = [[Point(1, 1), Point(1, 1)],
               [Point(1, 3), Point(1, 4)]]

        pts = np.array(pts)
        print("Representation format is (pt):hash")
        print(str(pts))
        assert pts[0][0] == pts[0][1], "Point's should be equal"

        grid.gen_simplex_grid()
        print("Simplex Grid shape: {}".format(grid.grid.shape))

        count = 0
        for i in range(grid.grid.shape[0]):
            for j in range(grid.grid.shape[1]):
                count += len(grid.grid[i, j].adjacent_edges)
        print("{} total edges, {} unique edges, total/unique = {}, expect about 2".format(
            count, len(grid.edges), count/len(grid.edges)))

        # Testing intersection finding math
        position = np.array([0.9, 0.9])
        pos_frac = position - np.fix(position)
        pos_whole = position - pos_frac
        remainder = np.fmod(pos_frac, 1)
        idx = np.array((position-remainder)/1, dtype=int)

        print("IDXS: {}".format(idx))

        a, b = np.array([1, 1]), np.array([2, 4])
        print(dist(a, b))

        # testing actual intersection finding
        grid.add_snake(snake)
        intersections = grid.get_snake_intersections()
        print("Intersections, 6 expected, found {}".format(
            len(intersections[0])))
        print(intersections)
