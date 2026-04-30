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
    # TODO: Implement this method!
    # Hint: p'T * F * p = 0
    raise NotImplementedError
    

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
    # TODO: Implement this method!
    raise NotImplementedError


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
    # TODO: Implement this method!
    raise NotImplementedError


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
    # TODO: Implement this method!
    raise NotImplementedError


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
