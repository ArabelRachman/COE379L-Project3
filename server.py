from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the saved model at startup.
# Make sure that "model.h5" is in the same directory as this file (or adjust the path accordingly).
model = tf.keras.models.load_model("model_best.h5")

def preprocess_image(image_bytes):
    """
    Process the raw image bytes so that they can be fed into your model.
    Steps:
      - Open with Pillow.
      - Convert to RGB (ensuring 3 channels).
      - Resize to 128x128.
      - Convert to a numpy array of type float32 and normalize pixel values to [0,1].
      - Add a batch dimension.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError("Could not open image: " + str(e))
    
    # Convert to RGB if needed (your model expects 3-channel input)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize the image to the shape that your model expects
    image = image.resize((128, 128))
    
    # Convert the image to a numpy array and normalize pixel values
    image_array = np.array(image).astype("float32") / 255.0
    
    # Add the batch dimension: (1, 128, 128, 3)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route("/summary", methods=["GET"])
def model_summary():
    """
    Returns basic metadata about the model in JSON format.
    """
    metadata = {
        "model_type": "lenet_alternate", 
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "number_of_parameters": model.count_params()
    }
    return jsonify(metadata)

@app.route("/inference", methods=["POST"])
def inference():
    """
    Accepts an image file via a multi-part form under the key 'image'.
    Preprocesses the image, runs prediction on the model, and
    returns a JSON response with a top-level key 'prediction' having
    the value 'damage' if the model output > 0.5, or 'no_damage' otherwise.
    """
    # Check if the POST request has the file part.
    if "image" not in request.files:
        return jsonify({"error": "Missing file in request. Provide an image file under the key 'image'."}), 400
    
    file = request.files["image"]
    
    # Ensure that a file was uploaded.
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400
    
    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400

    # Perform inference. As this is a binary classification with sigmoid activation,
    # the model returns a probability. We threshold at 0.5.
    prediction_prob = model.predict(processed_image)[0][0]
    prediction_label = "" \
    "" if prediction_prob > 0.5 else "damage"
    
    return jsonify({"prediction": prediction_label})

if __name__ == "__main__":
    # Run the server on all available interfaces (0.0.0.0) on port 5000.
    app.run(host="0.0.0.0", port=5000, debug=True)
