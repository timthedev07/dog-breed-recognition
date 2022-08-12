import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import Sequential as SequentialType
from sklearn.model_selection import train_test_split
import termcolor as tc

csvFname = "labels.csv"

pathJoin = os.path.join

DATA_DIR = "data"
IMG_DIR = "images"

class DogBreedModel:
    def __init__(self, trainPercentage = 80, production = False, dataSize = None) -> None:
        """
        Params:
          - `trainPercentage`% indicates how much of the given data should be used for **training**
          - `dataSize` indicates how much of the downloaded data should be used, leave as None if all data should be involved
        """
        self.trainPercentage = trainPercentage
        self.BATCH_SIZE = 32
        self.dataSize = dataSize

        self.labels = []
        self.model: SequentialType = None
        self.RESIZED_IMG_WIDTH = 224
        self.RESIZED_IMG_HEIGHT = 224

        self.labelsData = pd.DataFrame()

        # train/test data
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []

        # storing the classifier
        self.classifier: SequentialType = Sequential([])

        self.populateLabels()
        self.initClassifier()

        if not production:
            self.loadDataset()
            self.initModel()
        else:
            self.loadModel()

    def getImgLabelPair(self, fileName: str, label):
        return self.imgToNp(fileName), label

    def createDataBatches(self, X, y=None, validation = False):
        if validation:
            validationData = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
            return validationData.map(self.getImgLabelPair).batch(self.BATCH_SIZE)
        else:
            trainData = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))).shuffle(len(X))
            return trainData.map(self.getImgLabelPair).batch(self.BATCH_SIZE)

    def initClassifier(self):
        """
        Initializing the pre-trained resnet 50 v2 model.

        Call this before other initialization methods.
        """
        classifier = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(self.RESIZED_IMG_WIDTH, self.RESIZED_IMG_HEIGHT, 3),
            include_top = False,
            weights='imagenet',
            classes=len(self.labels)
        )
        for layer in classifier.layers:
            layer.trainable = False

        self.classifier = classifier

        print(tc.colored("Classifier model initialized.", "green"))

    def populateLabels(self):
        labelsInfo = pd.read_csv(pathJoin(DATA_DIR, csvFname))
        breeds = labelsInfo["breed"].unique()
        breeds.sort()

        self.labels = breeds

        print(tc.colored("Labels populated.", "green"))

    def yDataOneHot(self, y: np.ndarray):
        """
        y should be a 1-dimensional array containing the breeds as strings
        """
        return np.array([(label == self.labels).astype(int) for label in y])

    def xNormalize(self, x: np.ndarray):
        """
        turns all pixel values in all images provided into a decimal in the range of [0, 1]
        """
        return tf.image.convert_image_dtype(x, tf.float32)

    def loadDataset(self):
        print(tc.colored("Loading dataset & labels...", "yellow"))

        csvData = pd.read_csv(pathJoin(DATA_DIR, csvFname))
        imgFilenames = csvData["id"]
        n = self.dataSize if self.dataSize is not None else len(imgFilenames)
        imgFilenames = imgFilenames[:n].map(lambda x: pathJoin(DATA_DIR, IMG_DIR, x + ".jpg"))
        print(tc.colored("  Image file names loaded.", "green"))

        allY = self.yDataOneHot(csvData["breed"].to_numpy()[:n])
        print(tc.colored("  Labels one-hot encoded.", "green"))

        self.trainX, self.testX, self.trainY, self.testY = train_test_split(imgFilenames, allY, test_size=(1 - (self.trainPercentage / 100)), shuffle=False)
        print(tc.colored("Dataset & labels loaded.", "green"))

    def imgToNp(self, fileName: tf.Tensor):
        """
        Reads a given image, resize it, normalize it, and converts it to a tf tensor

        The `fileName` param should be a string tensor
        """
        file = tf.io.read_file(fileName)
        img = tf.io.decode_image(file, channels = 3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_crop_or_pad(img, 224, 224)
        return img

    def initModel(self):
        """
        Initialized an untrained model defined with an architecture
        """
        self.model = Sequential([
            self.classifier,
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dropout(0.45),
            Dense(128),
            Dropout(0.45),
            Dense(len(self.labels), activation="softmax")
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print(tc.colored("Model compiled.", "green"))

    def trainModel(self):
        print(tc.colored("Creating data batches for training...", "yellow"))
        trainData = self.createDataBatches(self.trainX, self.trainY)
        validationData = self.createDataBatches(self.testX, self.testY, validation=True)
        print(tc.colored("Data batches created.", "green"))

        self.model.fit(
            trainData,
            steps_per_epoch = len(trainData),
            epochs = 15,
            validation_data = validationData,
            validation_steps = len(validationData)
        )
        self.model.evaluate(validationData)
        self.saveModel()

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


def main():
    model = DogBreedModel()
    model.trainModel()

if __name__ == "__main__":
    main()