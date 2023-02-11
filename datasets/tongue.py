import os
import cv2
import numpy as np
from PIL import Image


def get_data(path="./color"):
    cls2idx = {}
    idx2cls = {}
    y = []
    X = []
    for idx, folder in enumerate(os.listdir(path)):
        folderpath = os.path.join(path, folder)
        cls2idx[folder] = idx
        idx2cls[idx] = folder
        for file in os.listdir(folderpath):
            filepath = os.path.join(folderpath, file)
            image = Image.open(filepath)
            image = image.resize((180, 180))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            y.append(idx)
            X.append(image)
    return np.array(X), np.array(y), cls2idx, idx2cls
