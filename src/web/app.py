from src.backend.model import DogBreedModel
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tempfile
import os

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "webp"])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getTempFilePath(filename):
    return os.path.join(UPLOAD_FOLDER, secure_filename(filename))

def getImgUpload(method: str):
    if method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return {
                "message": "No image found"
            }, 400

        file = request.files["file"]
        fname = file.filename
        if not fname or fname == "":
            return {
                "message": "No file selected"
            }, 400

        if file and allowed_file(fname):
            path = getTempFilePath(fname)
            file.save(path)
            return path
        else:
            return {
                "message": "File extension not allowed"
            }, 400



@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        imgRes = getImgUpload(request.method)
        print(imgRes)

        if type(imgRes) != str:
            return imgRes

        model = DogBreedModel(production = True, breedTxtFilePath="breeds.txt")
        pred = model.predictPicture(imgRes)

        return {
            "breed": pred,
        }, 200
    else:
        return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)