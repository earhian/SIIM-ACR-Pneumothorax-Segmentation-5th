import random
from functools import partial
import pandas as pd
import cv2
import numpy as np
from torch.nn.parallel.data_parallel import data_parallel
import torch
import math
def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


random.seed(123)
EXDATAPATH = './input/images_*/images/*.png'

NUM_CLASSES = 1
SIZE = 1024
