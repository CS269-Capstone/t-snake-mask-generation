"""
Main entry point for the project.
"""
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

import tsnake.initialize as init
import tsnake.utils as utils
from tsnake.grid import Grid
# from jiahui.inpaint import inpaint as j_inpaint


class Main(object):
    """
    Class acting as the main entry point for the project.
    TODO: detect snake convergence
    Args:
    ======================================
    (str) path_to_image:
    * The path to the image (.png) file.
    (str) path_to_mask:
    * The path to the mask (.png) file.
    ======================================
    """

    def __init__(self, path_to_image, path_to_mask, ):
        self.path_to_image = path_to_image
        self.path_to_mask = path_to_mask

        # The original image
        self.grayscale_image = init.load_grayscale_image(path_to_image)
        self.color_image = cv2.imread(path_to_image)

        # The (messy) user-defined mask
        self.user_mask = init.load_mask(path_to_mask, convert=True)
        # The mask as cleaned up by the T-snake(s)
        self.snake_mask = None

        # The image inpainted using the (messy) hand-defined mask
        self.user_output_image = None
        # The image inpainted using the snake-defined mask
        self.snake_output_image = None

        # ========================================================
        # A list of MaskedRegion objects (see initialize.py)
        self.masked_regions = None
        # The Grid instance for the image being operated on
        # (self.grayscale_image, see grid.py)
        self.grid = None
        # A list of corresponding lists of Tsnake instances
        self.snakes = None

    def _inpaint(self, which_mask='snake'):
        valid_ = ['snake', 'user']
        assert which_mask in valid_, 'Please choose a mask from %s.' % valid_

        if which_mask == 'snake' and self.snake_mask is None:
            msg = 'T-snake has not been run yet - please call Main.run() first.'
            raise ValueError(msg)
        
        # inpainting no longer works :( 
#         if which_mask == 'user':
#             output = j_inpaint(self.color_image, self.user_mask)
#             self.user_output_image = output
#         else:
#             output = j_inpaint(self.color_image, self.snake_mask)
#             self.snake_output_image = output

        return output

    def _snakes_to_mask(self, snakes):
        """
        Take a list of snakes, and return an np array denoting a mask
        Args:
        =======================
        * snakes: list(TSnake), list of all converged tsnakes for the image
        =======================
        Return:
        =======================
        * mask: np.array containing the mask as 0s and 1s
        =======================
        """
        assert isinstance(self.grid, Grid), 'Grid should have been initialized, but was not'

        mask = np.zeros(self.grayscale_image.shape)
        simplex_grid = Grid(image=self.grayscale_image, scale=1)
        simplex_grid.gen_simplex_grid()
        _ , grid_node_queues = simplex_grid.reparameterize_phase_one(snakes)
        simplex_grid.reparameterize_phase_two(
            snakes=snakes, grid_node_queues=grid_node_queues)
        grid = simplex_grid.grid

        n, m = mask.shape
        for i in range(n):
            for j in range(m):
                mask[i, j] = 255 if grid[i, j].is_on else 0

        total_num_nodes = n * m
        print("Main found [{}/{}] nodes to be on".format(are_on, total_num_nodes))
        return mask

    def run(self, max_iter=1000, grid_scale=1.0, tolerance=0.5, **snake_params):
        """
        Args:
        ======================================
        (int) max_iter:
        * The number of snake-update iterations to perform before giving up.
        (float) grid_scale:
        * The argument `scale` to be passed to the `Grid` constructor.
        (float) tolerance:
        * A float `tolerance > 0` such that if a T-snake's nodes each move by
          less than `tolerance`, then the T-snake is considered to have converged.
        ======================================
        """
        assert max_iter > 0
        assert grid_scale > 0
        assert tolerance > 0

        M = snake_params.pop('M', 5)

        # ====================================================
        # INITIALIZATION =====================================
        # ====================================================
        self.masked_regions = init.compute_masked_regions(
            self.grayscale_image, self.user_mask
        )
        # One grid per image
        self.grid = Grid(self.grayscale_image, scale=grid_scale)
        self.snakes = [None] * len(self.masked_regions)

        for r_num, region in enumerate(self.masked_regions):
            # Initialize T-snake from each masked region
            t_snake = region.initialize_tsnake(**snake_params)
            self.snakes[r_num] = [t_snake]

        # ====================================================
        # RUN THE T-SNAKE ====================================
        # ====================================================
        # set of indices of MaskedRegions whose T-snake(s) have not converged yet
        to_finish = set(list(range(len(self.masked_regions))))
        iter_num = 0
        while len(to_finish) > 0 and iter_num < max_iter:
            to_discard = set()
            
            for r_num in to_finish:
                grid = self.grid
                snakes = self.snakes[r_num]

                # Run:
                # 1) the m-step function (which has M deformation steps), and
                # 2) the reparameterization (occuring every M deformation steps)
                for snake in snakes:
                    snake.m_step(M)
                new_snakes, _ = grid.reparameterize_phase_one(snakes)

                # =======================================
                # CHECK FOR CONVERGENCE =================
                # =======================================
                # NECESSARY CONDITION #1: the number of T-snakes in this
                # MaskedRegion remained constant (no snakes have merged or split)
                const_num_snakes = len(new_snakes) == len(snakes)
                if not const_num_snakes:
                    self.snakes[r_num] = new_snakes
                    continue

                # NECESSARY CONDITION #2: for every snake, the number of
                # nodes per T-snake remained constant
                const_nodes_per_snake = all([snakes[i].num_nodes == new_snakes[i].num_nodes for i in range(len(snakes))])
                if not const_nodes_per_snake:
                    self.snakes[r_num] = new_snakes
                    continue

                converged = True
                # NECESSARY CONDITION #3: for every snakes, each node's position
                # has moved by no more than `tolerance`
                for s in range(len(snakes)):
                    # check if any nodes have moved by more than `tolerance`
                    if any([utils.dist(snakes[s].nodes[i].position, new_snakes[s].nodes[i].position) > tolerance for i in range(len(snakes[s].nodes))]):
                        converged = False
                        break

                self.snakes[r_num] = new_snakes
                if converged:
                    to_discard.add(r_num)
                    
            to_finish = to_finish.difference(to_discard)
            iter_num += 1

        if len(to_finish) > 0:
            warnings.warn('One or more T-snakes did not converge: see MaskedRegions %s' % to_finish)

        # ====================================================
        # EXTRACT THE RESULTING IMAGE ========================
        # ====================================================
        all_snakes = []
        for snake_list in self.snakes:
            for s in snake_list:
                all_snakes.append(s)

        self.snake_mask = self._snakes_to_mask(all_snakes)
        # self.snake_output_image = self._inpaint('snake')
        # self.user_output_image = self._inpaint('user')
        return self.snake_mask

    def compare_masks(self, figsize=(15, 5)):
        # number of masked pixels
        npm_user = np.argwhere(self.user_mask != 0).shape[0]
        npm_snake = np.argwhere(self.snake_mask != 0).shape[0]
        
        pct_reduction = (npm_user - npm_snake) / npm_snake

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        f.suptitle('Percent reduction in masked pixels: %f' % pct_reduction)
        ax1.set_title('Original image')
        ax2.set_title('User-defined mask (NPM=%d)' % npm_user)
        ax3.set_title('Snake-defined mask (NPM=%d)' % npm_snake)

        # https://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/
        ax1.imshow(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        ax2.imshow(self.user_mask, cmap=plt.cm.binary)
        ax3.imshow(self.snake_mask, cmap=plt.cm.binary)
        
        plt.show()

        



    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    pass











