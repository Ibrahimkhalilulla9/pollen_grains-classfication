from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(__name__)
model = load_model("model1.h5")

# Class index mapping
index = [str(i) for i in range(21)]  # ['0', '1', ..., '20']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        image_path = os.path.join("static/uploads", image.filename)
        image.save(image_path)

        # Preprocess the image
        img = load_img(image_path, target_size=(128, 128))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict
        prediction = np.argmax(model.predict(x), axis=1)[0]
        result = index[prediction]

        return render_template("index.html", prediction=result, image_path=image_path)

    return render_template("index.html", prediction=None, image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
