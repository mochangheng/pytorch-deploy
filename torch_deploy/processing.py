import numpy as np
from PIL import Image

def list2PIL(l):
    return Image.fromarray(np.asarray(l, dtype=np.uint8))