import numpy as np

def roll_axis(img):
    img = np.rollaxis(img, -1, 0)
    img = np.rollaxis(img, -1, 0)
    return img