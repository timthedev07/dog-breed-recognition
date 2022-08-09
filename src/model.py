import pandas as pd
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from classification_models.tfkeras import Classifiers
from tensorflow.python.keras.models import Sequential as SequentialType
from tqdm import tqdm
import termcolor as tc

pathJoin = os.path.join

DATA_DIR = "data"
IMG_DIR = "images"

class DogBreedModel:
    def __init__(self, trainPercentage = 80) -> None:
        """
        Params:
          - `trainPercentage`% indicates how much of the given data should be used for training
        """
        self.trainPercentage = trainPercentage

        # this maps a breed to a number
        self.labels = {}
        self.model: SequentialType = None
        self.RESIZED_IMG_WIDTH = 256
        self.RESIZED_IMG_HEIGHT = 256

        self.labelsData = pd.DataFrame()

        # train/test data
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []

        # storing the classifier
        self.classifier = {
            "model": None,
            "preprocessInput": None,
        }

        self.initClassifier()
        self.populateLabels()
        self.loadDataset()
        self.initModel()

    def initClassifier(self):
        """
        Initializing the pre-trained resnet 34 model.

        Call this before other initialization methods.
        """
        ResNet34, preprocess_input = Classifiers.get('resnet34')
        self.classifier["model"] = ResNet34((self.RESIZED_IMG_WIDTH, self.RESIZED_IMG_WIDTH, 3), weights='imagenet')
        self.classifier["preprocessInput"] = preprocess_input

        print(tc.colored("Classifier model initialized.", "green"))

    def populateLabels(self):
        labelsInfo = pd.read_csv(pathJoin(DATA_DIR, "labels.csv"))
        breeds = labelsInfo["breed"].unique()
        breeds.sort()

        n = len(breeds)

        for i in range(n):
            self.labels[breeds[i]] = i + 1

        print(tc.colored("Labels populated.", "green"))

    def loadDataset(self):
        labelsInfo = pd.read_csv(pathJoin(DATA_DIR, "labels.csv"))
        dirContent = os.listdir(pathJoin(DATA_DIR, IMG_DIR))
        n = 100 # len(dirContent)
        dirContent.sort()

        allImages = []

        for fName in tqdm(dirContent[:n]):
            allImages.append(self.imgToNp(fName))

        allImages = np.array(allImages)

        splitPoint = int(n * self.trainPercentage / 100)

        self.trainX = allImages[:splitPoint]
        self.testX = allImages[splitPoint:]

        allY = labelsInfo["breed"].to_numpy()[:n]
        self.trainY = allY[:splitPoint]
        self.testY = allY[splitPoint:]

        labelBreed = lambda yData: np.array(list(map(lambda a: self.labels[a], yData)))

        numLabels = len(self.labels)
        # one hot encoding for y values for matching shapes
        self.trainY, self.testY = (tf.one_hot(labelBreed(self.trainY), numLabels),
                     tf.one_hot(labelBreed(self.testY), numLabels))

        print(tc.colored("Dataset & labels loaded.", "green"))

    def preprocessDataset(self):
        """
        Feature extraction for trainX and testX.

        Call this after `self.initClassifier()`
        """
        # f = lambda x: self.classifier["model"].predict(np.array(x))
        # self.trainX = f(self.trainX)
        # self.testX = f(self.testX)

        print(tc.colored("Feature extraction complete.", "green"))

    def imgToNp(self, fileName):
        """
        Reads a given image, resize it, and converts it to a numpy array
        """
        img = Image.open(pathJoin(DATA_DIR, IMG_DIR, fileName))
        img = img.resize((self.RESIZED_IMG_WIDTH, self.RESIZED_IMG_HEIGHT))
        npArr = np.asarray(img)
        return npArr

    def initModel(self):
        """
        Initialized an untrained model defined with an architecture
        """
        self.model = Sequential([
            self.classifier["model"],
            # Conv2D(32, (3, 3), activation='relu',
            #                    kernel_initializer='he_uniform', input_shape=(self.RESIZED_IMG_WIDTH, self.RESIZED_IMG_HEIGHT, 1)),
            # MaxPooling2D(pool_size=(2, 2)),
            # Flatten(),
            Dense(64, activation="relu", kernel_initializer="he_uniform"),
            Dropout(0.45),
            Dense(len(self.labels), activation="softmax")
        ])

        self.model.compile(
            optimizer="SGD",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print(tc.colored("Model compiled.", "green"))

    def trainModel(self):
        self.preprocessDataset()
        self.model.fit(self.trainX, self.trainY)

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
        [prediction] = self.model.predict([image])

        print(prediction)
        # for key, val in self.labels.values():
        #     if val == outputLabelNum:
        #         return key

        raise Exception("Unknown prediction.")


# def main():
model = DogBreedModel()

# if __name__ == "__main__":
#     main()