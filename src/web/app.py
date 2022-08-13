from curses import flash
from src.model import DogBreedModel
from flask import Flask, redirect, request, render_template
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

def uploadImg(method: str):
    if method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No image found')
            return redirect(request.url)

        file = request.files["file"]
        fname = file.filename
        if not fname or fname == "":
            flash('No file selected')
            return redirect(request.url)

        if file and allowed_file(fname):
            securedFName = secure_filename(fname)
            path = os.path.join(app.config['UPLOAD_FOLDER'], securedFName)
            file.save(path)
        else:
            flash('File extension not allowed.')
            return redirect(request.url)



@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = request.get_json()

        if not data["text"]:
            return "Bad Request", 400

        model = DogBreedModel(production = True, breedTxtFilePath="breeds.txt")
        res = model.predictPicture()

        return {
            "value": str(res),
            "sentiment": "negative" if res < 0 else ("positive" if res > 0 else "neutral")
        }, 200
    else:
        return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)