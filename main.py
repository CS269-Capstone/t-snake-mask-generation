import cv2
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

import tsnake.initialize as init
from tsnake.snake import TSnake, Element, Node
from tsnake.grid import Grid, Point
from tsnake.utils import dist, seg_intersect


if __name__ == '__main__':
    msk_path = 'examples/places2/case1_mask.png'
    img_path = 'examples/places2/case1_raw.png'
    mask = init.load_mask(msk_path)
    image = init.load_grayscale_image(img_path)

    regions = init._find_disjoint_masked_regions(mask)

    # NOTE: Uncomment to visialize initial masked regions
    # init.visualize_masked_regions(mask, regions)

    regions = init.compute_masked_regions(image, mask)
    tsnakes = []

    ### Parameters for T-Snakes ###
    sigma = 20.0  # gaussian filter sigma
    p = 1.0       # scale final image force with p
    c = 2.0       # scale gradient magnitude of image (applied before p)
    a = 1.0       # tension parameter
    b = 1.0       # bending parameter
    q = 1.0       # [todo] parameter
    gamma = 1.0   # friction coefficient
    dt = 1.0      # time step

    for region in regions:
        tsnake = region.initialize_tsnake(
            N=1000, p=p, c=c, sigma=sigma, a=a, b=b, q=q, gamma=gamma,
            dt=dt
        )
        tsnakes.append(tsnake)
        # region.visualize() # NOTE: To show tsnakes on images, uncomment

    tsnakes.sort(key=lambda t: len(t.nodes))
    print('length of T-Snakes (sorted) initialized on image:\n{}'.format(
        [len(t.nodes) for t in tsnakes]))

    image = init.load_grayscale_image(img_path)
    print('image shape:', image.shape)
    grid = Grid(image=image, scale=1)

    # Update grid
    # NOTE: Uncomment for force, expensive calculation
    force = grid.get_image_force(2,2,2)
    grid.gen_simplex_grid()
    print('simplex grid shape: {}'.format(grid.grid.shape))

    # Add snakes to grid
    for s in tsnakes:
        grid.add_snake(s)
