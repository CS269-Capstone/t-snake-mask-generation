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

    ### Parameters for TSnakes ### 
    sigma = 20.0  # gaussian filter sigma
    p = 1.0       # scale final image force with p
    c = 2.0       # scale gradient magnitude of image (applied before p)
    a = 1.0       # tension parameter
    b = 1.0       # bending parameter
    gamma = 1.0   # friction coefficient
    dt = 1.0      # time step
    
    for region in regions:
        tsnake = region.initialize_tsnake(
            N=1000, p=p, c=c, sigma=sigma, a=a, b=b, gamma=gamma,
            dt=dt
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
    force = grid.get_image_force(2,2,2)
    grid.gen_simplex_grid()
    print("Simplex Grid shape: {}".format(grid.grid.shape))

    # Add snakes to grid
    for s in tsnakes:
        grid.add_snake(s)
        break

    # Compute snake intersections with grid
    intersections = grid.get_snake_intersections()
    print(intersections)

    # Test snake evolution
    M = 20 #number of m-steps (iterations)
    snake = tsnakes[-1]

    # X and Y are matrices to save the position of the 
    # snake for every iteration
    X = np.zeros((snake.num_nodes,M+1))
    Y = np.zeros((snake.num_nodes,M+1))

    #save the initial position of the snake
    for i in range(snake.num_nodes):
        pos = snake.nodes[i].position
        X[i][0] = pos[0][0]
        Y[i][0]= pos[0][1]

    #run for M steps
    for j in range(M):
        #print(out[200,:,0])
        snake.m_step(1)

        #save the updated positions of the nodes
        for i in range(snake.num_nodes):
            pos = snake.nodes[i].position
            #print(pos)
            X[i][j+1] = pos[0][0]
            Y[i][j+1]= pos[0][1]


   