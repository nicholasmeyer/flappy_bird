import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt


def preprocess_frame(frame, y):
    # frame = frame[::5, 154:-154:5, :]
    a = y - 40
    b = y + 40
    frame = imresize(frame, (80, 80, 3))
    frame = np.mean(
        frame - np.array([144, 72, 17]), axis=2).astype('float32') / 255.
    frame[frame <= 0.36] = 0.0
    frame[frame > 0.36] = 1.0
    return frame
