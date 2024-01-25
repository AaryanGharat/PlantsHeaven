# app.py

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import keras
from keras.applications.vgg19 import preprocess_input

app = Flask(__name__, static_url_path='/static', template_folder='templates')

# Load the trained model
# give your saved model location
                                    
model = keras.models.load_model("archive/best_model1.h5")

# Class labels mapping
ref = {
    0: 'Apple Scab',
    1: 'Apple Blackrot',
    2: 'Apple Cedarapple Rust',
    3: 'Apple Healthy',
    4: 'Blueberry Healthy',
    5: 'Cherry Powdery Mildew',
    6: 'Cherry Healthy',
    7: 'Corn Cercospora Leaf Spot',
    8: 'Corn Common Rust',
    9: 'Corn Northern Leaf Blight',
    10: 'Corn (Maize) Healthy',
    11: 'Grape Black Rot',
    12: 'Grape Esca (Black Measles)',
    13: 'Grape Leaf Blight (Isariopsis Leaf Spot)',
    14: 'Grape Healthy',
    15: 'Orange Huanglongbing (Citrus Greening)',
    16: 'Peach Bacterial Spot',
    17: 'Peach Healthy',
    18: 'Pepper Bell Bacterial Spot',
    19: 'Pepper Bell Healthy',
    20: 'Potato Early Blight',
    21: 'Potato Late Blight',
    22: 'Potato Healthy',
    23: 'Raspberry Healthy',
    24: 'Soybean Healthy',
    25: 'Squash Powdery Mildew',
    26: 'Strawberry Leaf Scorch',
    27: 'Strawberry Healthy',
    28: 'Tomato Bacterial Spot',
    29: 'Tomato Early Blight',
    30: 'Tomato Late Blight',
    31: 'Tomato Leaf Mold',
    32: 'Tomato Septorial Leaf Spot',
    33: 'Tomato Two-Spotted Spider Mite',
    34: 'Tomato Target Spot',
    35: 'Tomato Yellow Leaf Curl Virus',
    36: 'Tomato Mosaic Virus',
    37: 'Tomato Healthy'
}


def predict_image_class(image_path):
    try:
        # Open the image using PIL
        img = Image.open(image_path)

        # Resize the image to the target size (255x255)
        img = img.resize((255, 255))

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Preprocess the input image for the model
        img_array = preprocess_input(img_array)

        # Expand dimensions to match the model's input shape
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction using the provided model
        prediction = model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(prediction)

        # Get the predicted class label
        predicted_class_label = ref[predicted_class_index]

        return jsonify({"prediction": predicted_class_label})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    if "image" in request.files:
        image = request.files["image"]
        if image.filename != "":
            prediction_result = predict_image_class(image)
            return prediction_result
        else:
            return jsonify({"error": "No file selected."})
    return jsonify({"error": "Invalid request."})










if __name__ == "__main__":
    app.run(debug=True)
