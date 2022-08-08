"""
This program converts raw images downloaded from:
https://www.kaggle.com/competitions/dog-breed-identification/data

to .npy files to speed up reading

Command line usage:
    python src/data-convert.py [--train|test]
"""

from typing import Literal, Union
from PIL import Image
import numpy as np
import os
import sys

from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_DATA_DIR = "npy-data"

pathJoin = os.path.join

def main():
    args = sys.argv[1:]
    n = len(args)

    isTrainingData = False

    if n > 0 and "test" in args[0]:
        isTrainingData = False
    else:
        isTrainingData = True

    ds = "train" if isTrainingData else "test"

    dirContent = os.listdir(pathJoin(DATA_DIR, ds))

    allImages = []

    for fName in tqdm(dirContent):
        allImages.append(imgToNp(fName, ds))

    allImages = np.array(allImages, dtype=object)

    np.save(pathJoin(OUTPUT_DATA_DIR, ds + ".npy"), allImages)


def imgToNp(fileName, dataSource: Literal["train", "test"]):
    img = Image.open(pathJoin(DATA_DIR, dataSource, fileName))
    npArr = np.asarray(img)
    return npArr

def getFile(imgId):
    return imgId + ".jpg"

if __name__ == "__main__":
    main()