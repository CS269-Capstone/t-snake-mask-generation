import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

import tsnake.initialize as init
from tsnake.snake import TSnake, Element, Node
from tsnake.grid import Grid, Point
from tsnake.utils import dist, seg_intersect

if __name__ == '__main__':
    msk_path = './examples/places2/case1_mask.png'
    img_path = './examples/places2/case1_raw.png'
    mask = init.load_mask(path=msk_path, convert=True)
    image = init.load_grayscale_image(img_path)

    regions = init._find_disjoint_masked_regions(mask)
    # NOTE: Uncomment to visialize initial masked reigons
    # init.visualize_masked_regions(mask, regions)
    regions = init.compute_masked_regions(image, mask)
    print('number of masked regions:', len(regions))

    tsnakes = []
    ### Parameters for T-snakes ###
    sigma = 20.0    # gaussian filter sigma
    p = 1.0         # scale final image force with p
    c = 2.0         # scale gradient magnitude of image (applied before p)
    a = 1.0         # tension parameter
    b = 1.0         # bending parameter
    q = 1.0         # amplitude of the inflation force
    gamma = 1.0     # friction coefficient
    dt = 1.0        # time step
    threshold = 10  # inflation force treshold
    for region in regions:
        tsnake = region.initialize_tsnake(
            N=1000, p=p, c=c, sigma=sigma, a=a, b=b, q=q, gamma=gamma,
            dt=dt, threshold=threshold
        )
        tsnakes.append(tsnake)
        # region.visualize() # NOTE: To show tsnakes on images, uncomment

    tsnakes.sort(key=lambda t: len(t.nodes))
    t_snake_lengths = [len(t.nodes) for t in tsnakes]
    print('Length of T-Snakes initialized on image:\n{}'.format(t_snake_lengths))

    image = init.load_grayscale_image(img_path)
    print('image shape: ', image.shape)
    grid = Grid(image=image, scale=1)

    # Update grid
    # NOTE: Uncomment for force, expensive calculation
    # force = grid.get_image_force(2,2,2)
    grid.gen_simplex_grid()
    print('Simplex grid shape: {}'.format(grid.grid.shape))

    print('shape of tsnakes:', np.shape(tsnakes))
    # Compute snake intersections with grid
    intersections = grid.get_snake_intersections(tsnakes)
    print('intersections shape:', np.shape(intersections))
    n_inter_for_each_t_snake = [len(inter) for inter in intersections]
    print('num of intersections for each t-snake:', n_inter_for_each_t_snake)

    # ----------------- Test T-Snake Evolution and Reparam -----------------
    regions[1].show_snake(save_fig='images/img0.png')
    iterations = 1 # dummy value for testing purposes
    M = 5
    # Pick just one (for testing purposes) of the snakes on the grid
    snakes = [tsnakes[1]]

    # run iterations=20 of:
    # 1) the m-step function (which has M=5 deformation steps), and
    # 2) the reparameterization (occuring every M=5 deformation steps)
    for j in tqdm(range(iterations)):
        for snake in snakes:
            snake.m_step(M)
        new_snakes = grid.reparameterize_phase_one(snakes)

        regions[1].show_snake(save_fig='images/img{}.png'.format(j+1))
