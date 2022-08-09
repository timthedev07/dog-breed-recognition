import pandas as pd
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import Sequential as SequentialType
from tqdm import tqdm
import termcolor as tc

pathJoin = os.path.join

DATA_DIR = "data"

class DogBreedModel:
    def __init__(self) -> None:
        # this maps a breed to a number
        self.labels = {}
        self.model: SequentialType = None
        self.IMG_WIDTH = 128
        self.IMG_HEIGHT = 128

        self.populateLabels()

    def populateLabels(self):
        labelsInfo = pd.read_csv(pathJoin(DATA_DIR, "labels.csv"))
        breeds = labelsInfo["breed"].unique()
        breeds.sort()

        n = len(breeds)

        for i in range(n):
            self.labels[breeds[i]] = i + 1
        print(tc.colored("Labels populated.", "green"))

    def initModel(self):
        """
        Initialized an untrained model defined with an architecture
        """
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu',
                               kernel_initializer='he_uniform', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(100, activation="relu",
                                kernel_initializer="he_uniform"),
            Dense(len(self.labels), activation="softmax")
        ])

    def trainModel(self):
        pass

    def loadModel(self, path = "model"):
        """
        When an instance is first initialized, use this method to load
        an entire model already saved somewhere(if applicable).

        `path`: path to the directory of the saved model
        """
        loaded: SequentialType = tf.keras.models.load_model(path)
        self.model = loaded
        print(tc.colored("Model loaded.", "green"))

    def saveModel(self, path = "model"):
        """
        Save a trained model
        """
        self.model.save(path)
        print(tc.colored("Model saved.", "green"))

    def predict(self, image: np.ndarray) -> str:
        [[outputLabelNum]] = self.model.predict([image])
        for key, val in self.labels.values():
            if val == outputLabelNum:
                return key

        raise Exception("Unknown prediction.")


def main():
    model = DogBreedModel()


if __name__ == "__main__":
    main()