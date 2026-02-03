#
#
#      0===========================0
#      |    MAREVA 3D Modelling    |
#      0===========================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Script of the practical session. Plane detection by RANSAC
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 19/09/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):

    
    point = np.zeros((3,1))
    normal = np.zeros((3,1))
    
    # TODO
    n = np.cross(points[1] - points[0], points[2] - points[0])
    
    n = n / np.linalg.norm(n)
    normal = n.reshape(3,1)
    point = points[0].reshape(3,1)

    return point, normal


def in_plane(points, ref_pt, normal, threshold_in=0.1):
    
    indices = np.zeros(len(points), dtype=bool)
    
    # TODO: return a boolean mask of points in range
    dist = np.abs((points - ref_pt.T).dot(normal))
    for i in range(len(points)):
        if dist[i] < threshold_in:
            indices[i] = True
    return indices


def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):
    
    best_ref_pt = np.zeros((3,1))
    best_normal = np.zeros((3,1))
    # TODO:
    points_indices = np.arange(len(points))
    max_inliers = 0
    N = len(points) 
    for _ in range(NB_RANDOM_DRAWS):
        sample_indices = np.random.choice(N, 3, replace=False)
        sample_points = points[sample_indices]
        ref_pt, normal = compute_plane(sample_points)
        inliers_mask = in_plane(points, ref_pt, normal, threshold_in)
        num_inliers = np.sum(inliers_mask)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_ref_pt = ref_pt
            best_normal = normal
    k = np.log(1 - 0.99) / np.log(1 - (max_inliers / N) ** 3)
    print(f"New best plane with {max_inliers} inliers.")
    print(f"Estimated number of iterations for 99% confidence: {int(np.ceil(k))}")
    return best_ref_pt, best_normal


def multi_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):
    
    plane_inds = np.zeros((1,))
    remaining_inds = np.zeros((1,))
    plane_labels = np.zeros((1,))
    points_indices = np.arange(len(points))

    # TODO:
    current_points = points.copy()
    for i in range(NB_PLANES):
        N = len(current_points) 
        max_inliers = 0
        best_ref_pt = np.zeros((3,1))
        best_normal = np.zeros((3,1))
        for _ in range(NB_RANDOM_DRAWS):
            sample_indices = np.random.choice(N, 3, replace=False)
            sample_points = points[sample_indices]
            ref_pt, normal = compute_plane(sample_points)
            inliers_mask = in_plane(current_points, ref_pt, normal, threshold_in)
            num_inliers = np.sum(inliers_mask)
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_ref_pt = ref_pt
                best_normal = normal
        best_inliers_mask = in_plane(current_points, best_ref_pt, best_normal, threshold_in)
        best_indices = points_indices[best_inliers_mask]
        best_labels = np.full(len(best_indices),i)
        plane_inds = np.concatenate((plane_inds, best_indices))
        plane_labels =  np.concatenate((plane_labels, best_labels))
        current_points = current_points[~best_inliers_mask]
        points_indices = points_indices[~best_inliers_mask]
        print(f"Extracted plane with {max_inliers} inliers.")

        remaining_inds = points_indices.astype(np.int32)
        plane_inds = plane_inds[1:].astype(np.int32)
        plane_labels = plane_labels[1:].astype(np.int32)

    
    return plane_inds, remaining_inds, plane_labels


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    N = len(points)

    # Computes the plane passing through 3 randomly chosen points
    # ***********************************************************
    #

    if True:

        # Define parameter
        threshold_in = 0.1

        # Take randomly three points
        pts = points[np.random.randint(0, N, size=3)]

        # Computes the plane passing through the 3 points
        t0 = time.time()
        ref_pt, normal = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]

        # Save the 3 points and their corresponding plane for verification
        pts_clr = np.zeros_like(pts)
        pts_clr[:, 0] = 1.0
        write_ply('../triplet.ply',
                  [pts, pts_clr],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../triplet_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #

    if True:

        # Define parameters of RANSAC
        NB_RANDOM_DRAWS = 100
        threshold_in = 0.05

        # Find best plane by RANSAC
        t0 = time.time()
        best_ref_pt, best_normal = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        points_in_plane = in_plane(points, best_ref_pt, best_normal, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]

        # Save the best extracted plane and remaining points
        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Find multiple planes in the cloud
    # *********************************
    #

    if True:

        # Define parameters of multi_RANSAC
        NB_RANDOM_DRAWS = 200
        threshold_in = 0.05
        NB_PLANES = 5

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
        t1 = time.time()
        print('\nmulti RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels.astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
