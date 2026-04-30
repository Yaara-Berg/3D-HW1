import os
import sys
import env
import src.utils.utils as utils

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def get_3D_object_points(chessboard_size: tuple) -> np.ndarray:
    """
    Get the 3D object points of a chessboard
    Args:
        chessboard_size: Tuple containing the number of columns and rows in the chessboard
    Returns:
        Numpy array containing the 3D object points
    """
    # TODO: Implement this method!
    raise NotImplementedError


def undistort_image(image: np.ndarray, 
                    camera_matrix: np.ndarray, 
                    dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Undistort an image
    Args:
        image: Numpy array containing the image
        camera_matrix: Numpy array containing the camera matrix
        dist_coeffs: Numpy array containing the distortion coefficients
    Returns:
        Numpy array containing the undistorted image
    """
    # TODO: Implement this method!
    # HINT: use scipy.ndimage.map_coordinates to remap the image
    raise NotImplementedError

def load_grayscale_image(image: np.ndarray) -> np.ndarray:
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def calibrate_camera(object_points: np.ndarray, 
                     corners: np.ndarray, 
                     image_size: tuple) -> tuple:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [object_points], [corners], image_size, None, None
    )

    return camera_matrix, dist_coeffs


def find_chessboard_corners(image: np.ndarray, chessboard_size: tuple) -> np.ndarray:
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)

    if ret is False:
        raise ValueError("Verify correct dimensions of chessboard")
    
    return corners


def refine_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)

    return corners


def draw_corners(image: np.ndarray, chessboard_size: tuple, corners: np.ndarray):
    cv2.drawChessboardCorners(image, chessboard_size, corners, True)
    plt.imshow(image)
    plt.title("Chessboard Corners")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(env.p4.output):
        os.makedirs(env.p4.output)  
    expected_camera_matrix = np.load(env.p4.expected_camera_matrix)
    expected_dist_coeffs = np.load(env.p4.expected_dist_coeffs)
    # Part 4.a
    ideal_intrinsic_matrix = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])

    # Part 4.b
    chessboard_size = (14, 9)  # (columns, rows)
    
    image = utils.load_image(env.p3.chessboard_path)
    grayscale_image = load_grayscale_image(image)
    corners = find_chessboard_corners(grayscale_image, chessboard_size)
    corners = refine_corners(grayscale_image, corners)
    draw_corners(image, chessboard_size, corners)
    Image.fromarray(image).save(env.p4.chessboard_corners)

    # Part 4.c
    object_points = get_3D_object_points(chessboard_size)
    camera_matrix, dist_coeffs = calibrate_camera(object_points, corners, grayscale_image.shape[::-1])
    print("Camera Matrix:")
    print(camera_matrix)
    assert np.allclose(camera_matrix, expected_camera_matrix, atol=1e-2), f"Camera matrix does not match this expected matrix:\n{expected_camera_matrix}"
    np.save(env.p4.camera_matrix, camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    assert np.allclose(dist_coeffs, expected_dist_coeffs, atol=1e-2), f"Distortion coefficients do not match these expected coefficients:\n{expected_dist_coeffs}"
    np.save(env.p4.dist_coeff, dist_coeffs)

    # Part 4.d
    undistorted_image = undistort_image(image, camera_matrix, dist_coeffs)
    plt.imshow(undistorted_image)
    plt.title("Undistorted Image")
    plt.show()
    Image.fromarray(undistorted_image).save(env.p4.undistorted_image)
