from flask import Flask, render_template, request
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.python.keras import Sequential as ST
import string
import re

SAVED_MODEL_DIR = "model"

app = Flask(__name__)

@tf.function
def custom_standardization(input_data):
    stop_words = set(stopwords.words('english'))

    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    stripped_html = tf.strings.regex_replace(stripped_html,r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', ' ')
    stripped_html = tf.strings.regex_replace(stripped_html, r'@([A-Za-z0-9_]+)', ' ' )
    for i in stop_words:
        stripped_html = tf.strings.regex_replace(stripped_html, f' {i} ', " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )

def loadModel() -> ST:
    custom_objects = {"custom_standardization": custom_standardization}
    with tf.keras.utils.custom_object_scope(custom_objects):
        loaded = tf.keras.models.load_model(SAVED_MODEL_DIR)
        return loaded


@app.route("/", methods = ["POST", "GET"])
def home():
    if request.method == "POST":
        data = request.get_json()

        if not data["text"]:
            return "Bad Request", 400

        model = loadModel()
        [[res]] = model.predict([data["text"]])

        return {
            "value": str(res),
            "sentiment": "negative" if res < 0 else ("positive" if res > 0 else "neutral")
        }, 200
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run()
