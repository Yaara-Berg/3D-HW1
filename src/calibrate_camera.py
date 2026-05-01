import os
import sys
import env
import src.utils.utils as utils

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage


def get_3D_object_points(chessboard_size: tuple) -> np.ndarray:
    """
    Get the 3D object points of a chessboard
    Args:
        chessboard_size: Tuple containing the number of columns and rows in the chessboard
    Returns:
        Numpy array containing the 3D object points
    """
    columns, rows = chessboard_size
    object_points = np.zeros((columns * rows, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return object_points

def _pixel_to_normalized_coordinates(u: np.ndarray, v: np.ndarray, camera_matrix: np.ndarray
                                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert image pixel coordinates to normalized camera coordinates
    Args:
        u: Numpy array containing the image pixel coordinates
        v: Numpy array containing the image pixel coordinates
        camera_matrix: Numpy array containing the camera matrix
    Returns:
        Tuple containing the camera pixel coordinates
    """
    x = (u - camera_matrix[0, 2]) / camera_matrix[0, 0]
    y = (v - camera_matrix[1, 2]) / camera_matrix[1, 1]
    return x, y


def _normalized_coordinates_to_pixel(x: np.ndarray, y: np.ndarray, camera_matrix: np.ndarray
                                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert normalized camera coordinates to image pixel coordinates
    Args:
        x: Numpy array containing the camera pixel coordinates
        y: Numpy array containing the camera pixel coordinates
        camera_matrix: Numpy array containing the camera matrix
    Returns:
        Tuple containing the image pixel coordinates
    """
    u = x * camera_matrix[0, 0] + camera_matrix[0, 2]
    v = y * camera_matrix[1, 1] + camera_matrix[1, 2]
    return u, v


def _apply_distortion_to_normalized_coordinates(x: np.ndarray, y: np.ndarray, dist_coeffs: np.ndarray
                                                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply distortion to normalized camera coordinates
    Args:
        x: Numpy array containing the normalized camera coordinates
        y: Numpy array containing the normalized camera coordinates
        dist_coeffs: Numpy array containing the distortion coefficients
    Returns:
        Tuple containing the distorted normalized camera coordinates
    """
    k1, k2, p1, p2, k3 = dist_coeffs[0]
    r2 = x**2 + y**2
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    x_distorted = x * radial + 2*p1*x*y + p2*(r2 + 2*x**2)
    y_distorted = y * radial + 2*p2*x*y + p1*(r2 + 2*y**2)
    return x_distorted, y_distorted


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
    image_height, image_width = image.shape[:2]
    u_grid, v_grid = np.meshgrid(np.arange(image_width), np.arange(image_height))
    x_normalized, y_normalized = _pixel_to_normalized_coordinates(u_grid, v_grid, camera_matrix)
    x_distorted, y_distorted = _apply_distortion_to_normalized_coordinates(
        x_normalized, y_normalized, dist_coeffs
    )
    u_distorted, v_distorted = _normalized_coordinates_to_pixel(
        x_distorted, y_distorted, camera_matrix
    )

    source_coordinates = [v_distorted.flatten(), u_distorted.flatten()]
    undistorted_image = np.zeros_like(image)
    for channel_index in range(image.shape[2]):
        undistorted_channel = scipy.ndimage.map_coordinates(
            image[:, :, channel_index], source_coordinates, order=1
        ).reshape(image_height, image_width)
        undistorted_image[:, :, channel_index] = undistorted_channel
    return undistorted_image


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
    image = utils.load_image(env.p3.chessboard_path)
    original_image = image.copy()
    grayscale_image = load_grayscale_image(image)

    # Part 4.a
    image_height, image_width = grayscale_image.shape
    field_of_view_radians = np.deg2rad(45)
    focal_length = min(image_height, image_width) / (2 * np.tan(field_of_view_radians / 2))
    ideal_intrinsic_matrix = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1]
    ])

    # Part 4.b
    chessboard_size = (14, 9)  # (columns, rows)
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
