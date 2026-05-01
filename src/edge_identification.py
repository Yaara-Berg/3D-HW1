import os
import env

import numpy as np
from PIL import Image, ImageDraw

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

def find_contours(binary_image: np.ndarray, foreground: int=1) -> np.ndarray:
    """
    Find the boundaries of objects in a binary image.
    Args:
        binary_image: A binary image with objects as foreground.
        foreground: The value of the foreground pixels.
    Returns:
        A list of pixel coordinates that form the boundaries of the objects.
    """
    foreground_mask = binary_image == foreground
    padded_foreground_mask = np.pad(foreground_mask, pad_width=1, mode="constant", constant_values=False)

    neighbor_masks = [
        padded_foreground_mask[:-2, :-2],
        padded_foreground_mask[:-2, 1:-1],
        padded_foreground_mask[:-2, 2:],
        padded_foreground_mask[1:-1, :-2],
        padded_foreground_mask[1:-1, 2:],
        padded_foreground_mask[2:, :-2],
        padded_foreground_mask[2:, 1:-1],
        padded_foreground_mask[2:, 2:],
    ]
    has_background_neighbor = np.logical_not(np.logical_and.reduce(neighbor_masks))
    contour_mask = foreground_mask & has_background_neighbor
    return np.argwhere(contour_mask)


class ContourImage():
    def __init__(self, image: Image):
        self.image = image
        self.binarized_image = None

    def binarize(self, threshold=128) -> None:
        """
        Convert the image to a binary image.
        """
        grayscale_image = np.asarray(self.image.convert("L"))
        self.binarized_image = (grayscale_image < threshold).astype(np.uint8)

    def show(self) -> None:
        self.to_PIL().show()

    def fill_border(self):
        """
        Fill the border of the binarized image with zeros.
        """
        if self.binarized_image is None:
            raise ValueError("Image must be binarized before filling the border.")

        border_pixel_stack = []
        image_height, image_width = self.binarized_image.shape
        for column_index in range(image_width):
            border_pixel_stack.append((0, column_index))
            border_pixel_stack.append((image_height - 1, column_index))
        for row_index in range(image_height):
            border_pixel_stack.append((row_index, 0))
            border_pixel_stack.append((row_index, image_width - 1))

        while border_pixel_stack:
            row_index, column_index = border_pixel_stack.pop()
            if self.binarized_image[row_index, column_index] == 0:
                continue

            self.binarized_image[row_index, column_index] = 0
            for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor_row = row_index + row_offset
                neighbor_column = column_index + column_offset
                if 0 <= neighbor_row < image_height and 0 <= neighbor_column < image_width:
                    border_pixel_stack.append((neighbor_row, neighbor_column))

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
    if plt is not None:
        plt.imshow(result_img)
        plt.title("Chessboard Contours")
        plt.show()
