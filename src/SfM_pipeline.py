import os
import sys
import env
import src.utils.engine
import src.utils.utils as utils
import numpy as np

from src.calibrate_camera import *
from src.image_rectification import *
from src.threeD_reconstruction import *


def reprojection_error(camera_params, camera_indices, point_indices, observed_2d):
    """
    Compute the reprojection error for the bundle adjustment problem.
    """
    # TODO: Extra credit!
    return NotImplementedError

def bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, observed_2d):
    """
    Perform bundle adjustment.
    """
    # TODO: Extra credit!
    # Hint: Use scipy.optimize.least_squares and reprojection_error()
    return NotImplementedError


def main():
    if not os.path.exists(env.p8.output):
        os.makedirs(env.p8.output)
    chessboard_size = (16, 10)  # columns, rows
    images_folder = env.p8.statue_images
    chessboard_path = env.p7.chessboard

    camera_matrix, dist_coeffs = calibrate_camera_from_chessboard(chessboard_path, chessboard_size)

    # USE THIS CAMERA MATRIX FOR THE REST OF THE PIPELINE
    camera_matrix = np.eye(3)
    focal_length = 719.5459
    camera_matrix[0,0] = focal_length
    camera_matrix[1,1] = focal_length

    image_files = sorted([
        f for f in os.listdir(images_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    global_points_3D = []
    R_global_prev = np.eye(3, dtype=np.float64)
    t_global_prev = np.zeros((3,), dtype=np.float64)

    for i in range(len(image_files) - 1):
        # TODO: Implement this method!
        img_path1 = os.path.join(images_folder, image_files[i])
        img_path2 = os.path.join(images_folder, image_files[i+1])

        im1 = utils.load_image(img_path1)
        im2 = utils.load_image(img_path2)

        kp1, kp2, good_matches = find_matches(im1, im2)
        F, mask, pts1, pts2 = recover_fundamental_matrix(kp1, kp2, good_matches)
        inlier_pts1, inlier_pts2 = get_inliers(mask, pts1, pts2)
        E = compute_essential_matrix(camera_matrix, F)
        R_candidates, t_candidates = estimate_initial_RT(E)
        R_local, t_local = find_best_RT(R_candidates, t_candidates, inlier_pts1, inlier_pts2)
        P1_local = get_identity_projection_matrix(camera_matrix)
        P2_local = get_local_projection_matrix(camera_matrix, R_local, t_local)

        pts4D_h = cv2.triangulatePoints(P1_local, P2_local, inlier_pts1, inlier_pts2)
        pts3D_local = (pts4D_h[:3] / pts4D_h[3]).T  # shape: (N,3)

        R_global_curr = R_global_prev @ R_local
        t_global_curr = R_global_prev @ t_local + t_global_prev

        pts3D_global = (R_global_prev @ pts3D_local.T).T + t_global_prev

        global_points_3D.append(pts3D_global)

        R_global_prev = R_global_curr
        t_global_prev = t_global_curr

        print(f"Pair {i}->{i+1}: Triangulated {len(pts3D_local)} local points; "
              f"transformed them to global. Global camera i+1 pose:\nR=\n{R_global_prev}\nt={t_global_prev}\n")

    if global_points_3D:
        final_cloud = np.vstack(global_points_3D)
        print(f"Finished. Accumulated {final_cloud.shape[0]} points in one global frame.")
        show_points_matplotlib(final_cloud)
    else:
        print("No points were accumulated.")


if __name__ == '__main__':
    main()

        
        
