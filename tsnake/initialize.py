"""
Module for reading in image+mask, then initializing the T-snakes accordingly.

NOTE: we might need to add some kind of padding to the mask-rectangles?
"""
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from .snake import Node, TSnake
from .utils import img_force, img_inflation_force


# =====================================================
# FOR LOADING IMAGE AND MASK FILES
# =====================================================

def load_grayscale_image(path):
    """
    Args:
    ==============
    (str) path:
    * Path to *raw* image file.
    ==============
    
    Returns:
    2D np.array representing *grayscale* image.
    """
    path = os.path.abspath(path)
    image = cv2.imread(path, 0)
    assert image is not None, 'failed to load image at path=%s' % path
    
    # sanity checks
    assert image.max() <= 255
    assert image.min() >= 0
    assert len(image.shape) == 2
    
    return image


def load_mask(path, convert=False):
    """
    Args:
    ==============
    (str) path:
    * Path to mask file.
    ==============
    
    Returns:
    2D binary np.array representing the mask.
      - mask[i, j] == 1    --> means that pixel (i, j) is masked
    """
    path = os.path.abspath(path)
    image = cv2.imread(path)
    assert image is not None, 'failed to load mask at path=%s' % path
    
    # Some asserts to make sure the input image is as expected
    if not convert:
        # Mask should be binary 0/255
        assert set(np.unique(image)) == {0, 255}, set(np.unique(image))
    else:
        # If the mask is hand-made and not perfect
        image[image >= 128 ] = 255
        image[image < 128] = 0
    # Mask should be 3D
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    
    # throw away useless 3rd dimension
    image = image[:, :, 0]
    # replace 255 -> 1
    image = np.where(image == 0, 0, 1)
    
    return image

# =====================================================
# FOR EXTRACTING DISTINCT RECTANGULAR MASKED REGIONS
# =====================================================

class _Region(object):
    """
    A rectangular mask region.
    
    This class is used as an intermediate for building MaskedRegion objects. 
    """
        
    def __init__(self):
        self.coords = set()

        self.leftmost = np.inf
        self.rightmost = -np.inf
        self.uppermost = -np.inf
        self.bottommost = np.inf

    def _is_member(self, r, c):
        for r1 in range(r-1, r+2):
            for c1 in range(c-1, c+2):
                if (r1, c1) in self.coords:
                    return True

        return False

    def _add_coordinate(self, r, c):
        self.coords.add((r, c))

        self.leftmost = min(self.leftmost, c)
        self.rightmost = max(self.rightmost, c)
        self.uppermost = max(self.uppermost, r)
        self.bottommost = min(self.bottommost, r)

    def add_if_member(self, r, c):
        """ 
        If (r, c) is adjacent (including diagonally) to any coordinate
        in this region, then add it to the region and return True.
        Else return False.
        """
        if self._is_member(r, c):
            self._add_coordinate(r, c)
            return True
        
        return False

    @property
    def bounding_box(self):
        """ TODO: doc """
        l, r = self.leftmost, self.rightmost
        u, b = self.uppermost, self.bottommost
        
        return (l, r, u, b)
    
    @classmethod
    def merge(cls, reg1, reg2):
        coords = reg1.coords.union(reg1.coords, reg2.coords)
        leftmost = min(reg1.leftmost, reg2.leftmost)
        rightmost = max(reg1.rightmost, reg2.rightmost)
        uppermost = max(reg1.uppermost, reg2.uppermost)
        bottommost = min(reg1.bottommost, reg2.bottommost)

        region = _Region()
        region.coords = coords
        region.leftmost = leftmost
        region.rightmost = rightmost
        region.uppermost = uppermost
        region.bottommost = bottommost
        return region

    def extract_masked_image_portion(self, image):
        l, r, u, b = self.bounding_box
        return image[b:u, l:r]


def _find_disjoint_masked_regions(mask):
    """
    Helper function for compute_masked_regions()
    
    TODO: this example can explain this function
    
    0 0 1 1 0 1
    0 0 0 1 1 0
    0 0 0 0 0 0
    0 1 1 1 0 1
    1 1 0 1 0 0
    
    TODO: doc
    """
    # indices (i, j) where mask[i, j] == 1
    # sorted in ascending order by i, ties broken by j
    mask_idx = sorted(np.argwhere(mask == 1).squeeze().tolist())
    
    regions = set()
    for (i, j) in mask_idx:
        assigned_to = set()
        for region in regions:
            if region.add_if_member(i, j):
                assigned_to.add(region)
        
        # if coordinate not assigned to any region, initialize new region
        if len(assigned_to) == 0:
            new_region = _Region()
            new_region._add_coordinate(i, j)
            regions.add(new_region)
            
        # if coordinate has been assigned to 2+ regions,
        #  merge those regions together 
        elif len(assigned_to) >= 2:
            merged = _Region()
            for reg in assigned_to:
                merged = _Region.merge(merged, reg)
                
            regions = regions.difference(assigned_to)
            regions.add(merged)

    return regions


def compute_masked_regions(raw_image, raw_mask):
    """
    Args:
    =====================================================
    (np.array) raw_image:
    * The full, un-masked, grayscale image.
    
    (np.array) raw_mask:
    * The full mask (a binary numpy matrix).
    =====================================================
    """
    assert raw_image.shape == raw_mask.shape
    
    regions = _find_disjoint_masked_regions(raw_mask)
    
    out = []
    for region in regions:
        out.append(MaskedRegion(region, raw_image, raw_mask))
        
    return out

# =====================================================
# MaskedRegion
# =====================================================

class MaskedRegion(object):
    """
    A rectangular sub-region of the full image corresponding to one 
    distinct sub-mask (a masked region which does not touch any other
    masked regions).
    
    MaskedRegion initialization should happen *only* in function
    compute_masked_regions().
    
    Args:
    =====================================================
    (_Region) _region:
    * The intermediate _Region object.
    
    (np.array) raw_image_full:
    * The FULL, un-masked, grayscale image - NOT just the portion
      of the image inside this rectangle.
      
    (np.array) raw_mask_full:
    * The FULL mask (binary array) - NOT just the portion of the
      mask inside this rectangle. Note that:
      raw_mask_full[i, j] = 1    ===>    pixel (i, j) is masked
    =====================================================
    """
    
    def __init__(
        self, _region, raw_image_full, raw_mask_full
    ):
        l, r, u, b = _region.bounding_box
        
        self.top_row = b      # the coordinates of this rectangular
        self.bottom_row = u   # (partially-) masked region
        self.left_col = l     # 
        self.right_col = r    # 
        
        # raw_image_portion == intensity_grid
        self.raw_image_portion = raw_image_full[b:u, l:r]
        self.raw_mask_portion = raw_mask_full[b:u, l:r]
        
        # (for visualization only)
        self.raw_image = raw_image_full
        self.raw_mask = raw_mask_full
        
        self._initial_tsnake = None
    
    def __str__(self):
        t, b = self.top_row, self.bottom_row
        l, r = self.left_col, self.right_col
        return "MaskedRegion(rows=%d:%d, cols=%d:%d)" % (t, b, l, r)
    
    def __repr__(self):
        return self.__str__()
    
    def visualize(self, figsize=(20, 20)):
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        ax1.set_title('raw input mask')
        ax2.set_title('corresp. rect. mask-region')
        ax3.set_title('grayscale image portion')
        ax4.set_title('local mask')
        
        ax1.imshow(self.raw_mask, cmap=plt.cm.binary)
        
        top, bottom = self.top_row, self.bottom_row
        left, right = self.left_col, self.right_col
        im2 = np.zeros_like(self.raw_image)
        im2[top:bottom, left:right] = 1
        ax2.imshow(im2, cmap=plt.cm.binary)
        
        ax3.imshow(self.raw_image_portion, cmap=plt.cm.binary)
        
        ax4.imshow(self.raw_mask_portion, cmap=plt.cm.binary)
        
        # If T-snake initialized, show its initial configuration
        if self._initial_tsnake is not None:
            nodes = [[n.y, n.x] for n in self._initial_tsnake.nodes]
            nodes = np.array(nodes).reshape(len(nodes), 2)

            norms = np.array(
                [[n.normal[1], n.normal[0]] for n in self._initial_tsnake.nodes], 
                dtype=np.float32
            )
            # multiply outward normals s they're easier to see
            norms = norms.reshape(-1,2) * 6

            norms += nodes
            # How many terminal and initial nodes to show in different colors
            buffer = 5
            ax4.scatter(
                nodes[buffer:-buffer, 0], nodes[buffer:-buffer:, 1], c='red', 
                s=3, alpha=0.5
            )
            
            # Visualize normals, and initial nodes (white), terminal nodes (yellow)
            ax4.scatter(nodes[:buffer, 0], nodes[:buffer, 1], c='white', s=3, alpha=0.9)
            ax4.scatter(
                nodes[-buffer:, 0], nodes[-buffer:, 1], c='yellow', s=3, alpha=0.9
            )
            ax4.scatter(norms[:, 0], norms[:, 1], c='green', s=3, alpha=0.5)

            for i in range(len(nodes)):
                # Elements are plotted in red 
                ax4.plot(
                    nodes[[i-1,i], 0], nodes[[i-1,i], 1], c='red', lw=1, alpha=0.5
                )
                # Normals are plotted in green
                ax4.plot(
                    [nodes[i,0], norms[i,0]], [nodes[i,1], norms[i,1]], c='green', 
                    lw=1, alpha=0.5
                )
        
        plt.tight_layout()
        plt.show()
    
    def show_snake(self, save_fig='', figsize=(8, 8)):
        """
        Shows the current T-snake overlaid onto the (grayscale) image.
        """
        image = self.raw_image_portion
        if self._initial_tsnake is None:
            raise ValueError('T-snake has not been initialized.')
        
        snake = self._initial_tsnake
        # positions of the snake nodes
        nodes = [[n.y, n.x] for n in self._initial_tsnake.nodes]
        nodes = np.array(nodes).reshape(len(nodes), 2)
        # outward normals for each node
        norms = np.array(
            [[n.normal[1], n.normal[0]] for n in self._initial_tsnake.nodes], 
            dtype=np.float32
        )
        # multiply outward normals so they're easier to see
        norms = norms.reshape(-1,2) * 6
        norms += nodes
        
        f, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap=plt.cm.binary)
        
        # Plot nodes
        ax.scatter(nodes[:, 0], nodes[:, 1], c='red', s=3)
        # Plot endpoints of the outward normals
        ax.scatter(norms[:, 0], norms[:, 1], c='green', s=3, alpha=0.5)
        
        for i in range(len(nodes)):
            # Plot the elements
            ax.plot(
                nodes[[i-1,i], 0], nodes[[i-1,i], 1], c='red', lw=1, alpha=0.5
            )
            # Normals are plotted in green
            ax.plot(
                [nodes[i,0], norms[i,0]], [nodes[i,1], norms[i,1]], c='green', 
                lw=2, alpha=0.5
            )
        if save_fig == '':
            plt.show()
        else:
            plt.savefig(save_fig)
            plt.clf()
            
    def initialize_tsnake(
        self, N, p, c, sigma, a, b, q, gamma, dt, threshold=100, verbose=False
    ):
        """
        Initializes a T-snake by placing the initial nodes along 
        the boundary between masked and unmasked pixels.
        
        NOTE: we may eventually need to find a way to initialize 
              the nodes a little more intelligently
        
        Args:
        =====================================================
        (int) N:
        * The initial number of T-snake nodes.
        
        (float) p:
        * The hyperparameter p from Equation (7).
        
        (float) c:
        * The hyperparameter c from Equation (A.4).
        
        (float) sigma:
        * The hyperparameter sigma from Equation (A.4).
        
        (float) a:
        * The hyperparameter a from Equations (1), (8).
        
        (float) b:
        * The hyperparameter b from Equations (1), (8).

        (float) q:
        * The hyperparameter q from Equations (4).
        
        (float) gamma:
        * The hyperparameter gamma from Equations (1), (8).
        
        (bool) verbose:
        * Set to True to print extra information.
        =====================================================
        """
        # =============================================================
        # Argument checks =============================================
        # =============================================================
        assert N > 1, 'there must be at least 2 T-snake nodes (got N=%d).' % N
        assert p > 0, 'Hyperparameter p (Eq 7) must be > 0 (got p=%f).' % p
        assert c > 0, 'Hyperparameter c (Eq A.4) must be > 0 (got c=%f).' % c
        assert a > 0, 'Hyperparameter a (Eqs 1,8) must be > 0 (got a=%f).' % a
        assert b > 0, 'Hyperparameter b (Eqs 1,8) must be > 0 (got b=%f).' % b
        assert dt > 0, 'Hyperparameter dt (Eq 8) must be > 0 (got dt=%f).' % dt
        assert q > 0, 'Hyperparameter q (Eq 4) must be > 0 (got q=%f).' % q
        
        msg = 'Hyperparameter sigma (Eq A.4) must be > 0 (got sigma=%f).' % sigma
        assert sigma > 0, msg
        msg = 'Hyperparameter gamma (Eqs 1,8) must be > 0 (got gamma=%f).' % gamma
        assert gamma > 0, msg
        # =============================================================
        # =============================================================
        
        # pixels on the boundary between masked and unmasked
        edge_pixels = self._find_edge_pixels()

        # Pre-process sort to get right-handed spiral for normal init
        # This is necessary to get normal vectors to initialize properly
        edge_pixels.sort(key=lambda x:(x[0], x[1]), reverse=True)

        step = int(np.floor(len(edge_pixels) / N))
        step = max(step, 1)
        if verbose:
            print('Initializing a Node at every %d-th boundary-pixel' % step)
        
        # Order the nodes so that they connect to closest nodes
        ordered_nodes = self.order_snake_nodes(edge_pixels)
        
        # Pull out every step-th pixel for initializion as a Node
        nodes = []
        for node_num in range(0, len(ordered_nodes), step):
            r, c = ordered_nodes[node_num]
            new_node = Node(r, c)
            nodes.append(new_node)
            
        if verbose:
            print('Total of %d T-snake nodes were initialized' % len(nodes))
        
        # TODO@Cole: Replace hard-coded inflation force value, and
        # do these computations exclusively in the grid? Less 
        # memory overhead that way, and it makes more sense for subsequent steps
        force_grid = self.compute_force_grid(sigma, c, p)
        
        intensity_grid = img_inflation_force(self.raw_image_portion, threshold)
        
        snake = TSnake(
            nodes, force_grid, intensity_grid, a, b, q, gamma, dt
        )
        self._initial_tsnake = snake
        return snake
    
    def order_snake_nodes(self, node_locs):
        """
        TODO: doc
        """
        num_nodes = len(node_locs)
        node_locs = np.array(node_locs).reshape(num_nodes, 2)
        
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
            
        return ordered_nodes
    
    def compute_force_grid(self, sigma, c, p):
        """
        NOTE: can we manipulate the force grid to make sure the snake
              doesn't leave the user-masked area?
        
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
        out = img_force(self.raw_image_portion, sigma, c, p)
        return out
    
    def _find_edge_pixels(self):
        """
        Finds the pixels in this rectangular sub-region which are on the border
        between masked and unmasked.
        
        NOTE: returns a list of coordinates (row, col). These coordinates are
              relative to *this* rectangle's coordinate system, not the
              entire image's coordinate system.
              
        This naively does a check on each pixel, but efficiency here shouldn't
        matter.
        """
        # this rectangle's dimensions
        num_rows, num_cols = self.raw_mask_portion.shape
        
        out = []
        for r in range(num_rows):
            for c in range(num_cols):
                if self._is_edge_pixel(r, c):
                    out.append([r, c])
                    
        return out
    
    def _is_edge_pixel(self, r, c):
        """
        Returns True if the pixel at (r, c) (in this rectangle's coord system)
        is masked and also either adjacent to an unmasked pixel or on the edge
        of this rectangle.
        """
        # check this pixel is masked
        if not self.raw_mask_portion[r, c] == 1:
            return False
            
        for r1 in range(r-1, r+2):
            for c1 in range(c-1, c+2):
                if (r == r1) and (c == c1):
                    continue
                    
                # If either r1 or c1 is negative, then (r, c) is an edge pixel
                if (r1 < 0) or (c1 < 0):
                    return True
                    
                try:
                    if self.raw_mask_portion[r1, c1] == 0:
                        return True
                except IndexError:  # if IndexError, then this pixel is on the edge
                    return True
        
        return False

# =====================================================
# FOR VISUALIZATION
# =====================================================

def visualize_masked_regions(raw_mask, regions, figsize=None):
    """
    TODO: doc
    """
    total_masked = np.zeros_like(raw_mask)
    for region in regions:
        l, r, u, b = region.bounding_box
        total_masked[b:u, l:r] = 1
    
    if figsize is None:
        figsize = (10, 20)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.set_title('Input (raw) arbitrarily-shaped mask')
    ax2.set_title('Converted to disjoint rectangles')
    
    ax1.imshow(raw_mask)
    ax2.imshow(total_masked)
    plt.show()





    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    pass








