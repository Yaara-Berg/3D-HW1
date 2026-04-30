import numpy as np
from PIL import Image

def load_image(path):
    return np.array(Image.open(path))

def load_points(filename):
    '''
    GET_DATA_FROM_TXT_FILE
    Arguments:
        filename - a path (str) to the data location
    Returns:
        points - a matrix of points where each row is either:
            a) the homogenous coordinates (x,y,1) if the data is 2D
            b) the coordinates (x,y,z) if the data is 3D
        use_subset - use a predefined subset (this is hard coded for now)
    '''
    with open(filename) as f:
            lines = f.read().splitlines()
    number_pts = int(lines[0])

    points = np.ones((number_pts, 3))
    for i in range(number_pts):
        split_arr = lines[i+1].split()
        if len(split_arr) == 2:
            x, y = split_arr
        else:
            x, y, z = split_arr
            points[i,2] = z
        points[i,0] = x 
        points[i,1] = y
    return points