import os
import sys
import env
import src.utils.utils as utils

from PIL import Image

import numpy as np
import cv2
from src.fundamental_matrix import *
import matplotlib.pyplot as plt

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# NOTICE!! (I think the comment is wrong, because in main() it calculates F as p'Fp=0, so I will treat it that way)
def compute_epipole(points1: np.array, 
                    points2: np.array, 
                    F: np.array) -> np.array:
    '''
    Computes the epipole in homogenous coordinates
    given matching points in two images and the fundamental matrix
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
        F - the Fundamental matrix such that (points1)^T * F * points2 = 0

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        epipole - the homogenous coordinates [x y 1] of the epipole in the image
    '''
    lines = points1 @ F

    # Validate: each points2[i] must lie on its epipolar line lines[i].
    residuals = np.abs(np.sum(points2 * lines, axis=1))
    assert residuals.max() < 1.0, (
        f"Epipolar constraint badly violated: max |p2·l| = {residuals.max():.4f}"
    )
    
    _, _, Vt = np.linalg.svd(lines)
    e = Vt[-1]
    if np.abs(e[2]) > 1e-9:
        e = e / e[2]
    return e
    

def compute_matching_homographies(e2: np.array, 
                                  F: np.array, 
                                  im2: np.array, 
                                  points1: np.array, 
                                  points2: np.array) -> tuple:
    '''
    Determines homographies H1 and H2 such that they
    rectify a pair of images
    Arguments:
        e2 - the second epipole
        F - the Fundamental matrix
        im2 - the second image
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
    Returns:
        H1 - the homography associated with the first image
        H2 - the homography associated with the second image
    '''
    h, w = im2.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    # Center-translate / rotate / projective-send-to-infinity, then undo the
    # centering so the rectified images stay in the original pixel coordinate frame.
    T     = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    T_inv = np.array([[1, 0,  cx], [0, 1,  cy], [0, 0, 1]], dtype=np.float64)

    if np.abs(e2[2]) > 1e-9:
        e2_euc = e2 / e2[2]
        ex = e2_euc[0] - cx
        ey = e2_euc[1] - cy
        d = np.sqrt(ex**2 + ey**2)

        R = np.array([[ ex/d, ey/d, 0],
                      [-ey/d, ex/d, 0],
                      [    0,    0, 1]], dtype=np.float64)
        G = np.array([[1,    0, 0],
                      [0,    1, 0],
                      [-1/d, 0, 1]], dtype=np.float64)

        H2_basic = G @ R @ T
    else:
        H2_basic = np.eye(3)

    H2 = T_inv @ H2_basic

    # Transfer matrix M: maps image-1 points into the centered rectified frame.
    e2_cross = np.array([[    0, -e2[2],  e2[1]],
                         [ e2[2],     0, -e2[0]],
                         [-e2[1],  e2[0],     0]], dtype=np.float64)
    M = H2_basic @ e2_cross @ F   # H2_basic so third row equals expected H1[2]

    # Fit H1_basic row-by-row: H1_basic @ p1_i ≈ w_i * q2_i (centered frame).
    # Working in the original image-1 coordinate space avoids the near-constant
    # x-clustering that arises when using M-transformed coordinates.
    q2    = H2_basic @ points2.T;  q2    /= q2[2:3, :]   # centered rectified targets
    w_sc  = M[2] @ points1.T                              # projective denominator

    ABC, _, _, _ = np.linalg.lstsq(points1, q2[0] * w_sc, rcond=None)
    DEF, _, _, _ = np.linalg.lstsq(points1, q2[1] * w_sc, rcond=None)
    H1_basic = np.array([ABC, DEF, M[2]])

    H1 = T_inv @ H1_basic
    return H1, H2


def compute_rectified_image(im: np.array, 
                            H: np.array) -> tuple:
    '''
    Rectifies an image using a homography matrix
    Arguments:
        im - an image
        H - a homography matrix that rectifies the image
    Returns:
        new_image - a new image matrix after applying the homography
        offset - the offest in the image.
    '''
    from scipy.ndimage import map_coordinates

    h, w = im.shape[:2]

    # Warp the four corners to find the output canvas bounds
    corners = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]], dtype=np.float64)
    warped = H @ corners.T
    warped = warped / warped[2, :]

    min_x = int(np.floor(warped[0].min()))
    min_y = int(np.floor(warped[1].min()))
    max_x = int(np.ceil(warped[0].max()))
    max_y = int(np.ceil(warped[1].max()))
    new_w = max_x - min_x + 1
    new_h = max_y - min_y + 1

    # Build output pixel grid shifted by offset
    xs, ys = np.meshgrid(np.arange(new_w) + min_x, np.arange(new_h) + min_y)
    out_pts = np.stack([xs.ravel(), ys.ravel(), np.ones(new_w * new_h)], axis=0)

    # Map output pixels back to source via inverse H
    H_inv = np.linalg.inv(H)
    src = H_inv @ out_pts
    src = src / src[2:3, :]
    src_x = src[0].reshape(new_h, new_w)
    src_y = src[1].reshape(new_h, new_w)

    # Sample source image with bilinear interpolation
    if im.ndim == 3:
        channels = [
            map_coordinates(im[:, :, c].astype(np.float64), [src_y, src_x],
                            order=1, mode='constant', cval=0)
            for c in range(im.shape[2])
        ]
        new_image = np.stack(channels, axis=2).astype(im.dtype)
    else:
        new_image = map_coordinates(im.astype(np.float64), [src_y, src_x],
                                    order=1, mode='constant', cval=0).astype(im.dtype)

    return new_image, (min_x, min_y)


def find_matches(img1: np.array, img2: np.array) -> tuple:
    """
    Find matches between two images using SIFT
    Arguments:
        img1 - the first image
        img2 - the second image
    Returns:
        kp1 - the keypoints of the first image
        kp2 - the keypoints of the second image
        matches - the matches between the keypoints
    """
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return kp1, kp2, good_matches


def show_matches(img1: np.array, 
                 img2: np.array, 
                 kp1: list, 
                 kp2: list, 
                 matches: list) -> np.array:
    result_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.imshow(result_img)
    plt.title("SIFT Matches")
    plt.show()
    return result_img


if __name__ == '__main__':
    if not os.path.exists(env.p6.output):
        os.makedirs(env.p6.output)
    expected_e1, expected_e2 = np.load(env.p6.expected_e1), np.load(env.p6.expected_e2)
    expected_H1, expected_H2 = np.load(env.p6.expected_H1), np.load(env.p6.expected_H2)
    im1 = utils.load_image(env.p5.const_im1)
    im2 = utils.load_image(env.p5.const_im2)

    points1 = utils.load_points(env.p5.pts_1)
    points2 = utils.load_points(env.p5.pts_2)
    assert (points1.shape == points2.shape)
    F = normalized_eight_point_alg(points1, points2)

    # Part 6.a
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)
    assert np.allclose(e1, expected_e1, rtol=1e-2), f"e1 does not match this expected value:\n{expected_e1}"
    assert np.allclose(e2, expected_e2, rtol=1e-2), f"e2 does not match this expected value:\n{expected_e2}"
    np.save(env.p6.e1, e1)
    np.save(env.p6.e2, e2)

    # Part 6.b
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print
    print("H2:\n", H2)
    assert np.allclose(H1, expected_H1, rtol=1e-2), f"H1 does not match this expected value:\n{expected_H1}"
    assert np.allclose(H2, expected_H2, rtol=1e-2), f"H2 does not match this expected value:\n{expected_H2}"
    np.save(env.p6.H1, H1)
    np.save(env.p6.H2, H2)

    # Part 6.c
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)

    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)
    total_offset_y = np.mean(new_points1[:, 1] - new_points2[:, 1]).round()

    F_new = normalized_eight_point_alg(new_points1, new_points2)
    lines1 = compute_epipolar_lines(new_points2, F_new.T)
    lines2 = compute_epipolar_lines(new_points1, F_new)
    aligned_img = show_epipolar_imgs(rectified_im1, rectified_im2, lines1, lines2, new_points1, new_points2, offset=int(total_offset_y))
    Image.fromarray(aligned_img).save(env.p6.aligned_epipolar)

    # Part 6.d
    im1 = utils.load_image(env.p5.const_im1)
    im2 = utils.load_image(env.p5.const_im2)
    kp1, kp2, good_matches = find_matches(im1, im2)
    cv_matches = show_matches(im1, im2, kp1, kp2, good_matches)
    Image.fromarray(cv_matches).save(env.p6.cv_matches)
