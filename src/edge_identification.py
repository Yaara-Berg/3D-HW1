import os
import sys
import env
import src.utils.engine as engine

import numpy as np
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

def find_contours(binary_image: np.ndarray, foreground: int=1) -> np.ndarray:
    """
    Find the boundaries of objects in a binary image.
    Args:
        binary_image: A binary image with objects as foreground.
        foreground: The value of the foreground pixels.
    Returns:
        A list of pixel coordinates that form the boundaries of the objects.
    """
    # TODO: Implement this method!
    raise NotImplementedError


class ContourImage():
    def __init__(self, image: Image):
        self.image = image
        self.binarized_image = None

    def binarize(self, threshold=128) -> None:
        """
        Convert the image to a binary image.
        """
        # TODO: Implement this method!
        raise NotImplementedError

    def show(self) -> None:
        self.to_PIL().show()

    def fill_border(self):
        """
        Fill the border of the binarized image with zeros.
        """
        # TODO: Implement this method!
        raise NotImplementedError

    def to_PIL(self) -> Image:
        color_array = np.stack([self.binarized_image]*3, axis=-1) * 255
        color_array = color_array.astype(np.uint8)
        return Image.fromarray(color_array)
    
    def prepare(self) -> np.ndarray:
        self.binarize()
        self.fill_border()
        return self.binarized_image


def find_chessboard_contours(image: Image) -> np.ndarray:
    image = ContourImage(image)
    return find_contours(image.prepare())

def draw_corners(pil_img: Image, 
                 corners: np.ndarray, 
                 color: tuple=(255, 0, 0), 
                 radius: int=5) -> Image:
    img_with_corners = pil_img.copy()
    draw = ImageDraw.Draw(img_with_corners)
    
    for (y, x) in corners:
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], outline=color, width=2)
    
    return img_with_corners

if __name__ == "__main__":
    if not os.path.exists(env.p3.output):
        os.makedirs(env.p3.output)
    # engine.get_distorted_chessboard(env.p3.chessboard_path)

    image = Image.open(env.p3.chessboard_path)
    contours = find_chessboard_contours(image)

    result_img = draw_corners(image, contours, color=(255, 0, 0), radius=5)
    result_img.save(env.p3.contours_path)
    plt.imshow(result_img)
    plt.title("Chessboard Contours")
    plt.show()
