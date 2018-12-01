import numpy as np


def preprocess_frame(frame):
    frame = frame[::5, 154:-154:5, :]
    frame = np.mean(frame - np.array([144, 72, 17]), axis=2) / 255
    return frame
