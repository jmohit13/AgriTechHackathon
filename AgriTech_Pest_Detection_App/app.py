import os
import sys
import time
import numpy as np
from flask import (
    Flask,
    redirect,
    url_for,
    request,
    render_template,
    Response,
    jsonify,
    redirect,
)
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import (
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from util import base64_to_pil, load_labels

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":
        
        img = base64_to_pil(request.json)
        img.save("./assests/images/image.jpg")
        img = img.resize((width, height))

        input_data = np.expand_dims(img, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        top_k = results.argsort()[-5:][::-1]
        # labels = load_labels(label_file)
        for i in top_k:
            if floating_model:
                print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            else:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

        print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

        for i in top_k:
            top_label = results[i]
            top_label_prob = labels[i]
            break
        # print(top_label, top_label_prob)
        result = str(top_label_prob)  # Convert to string
        result = result.replace("_", " ").capitalize()
        return jsonify(result=result, probability=top_label_prob)

    return None


if __name__ == "__main__":

    # # Serve the app with gevent
    # http_server = WSGIServer(("0.0.0.0", 5000), app)
    # http_server.serve_forever()

    MODEL_PATH = "models/model_cpc_1.tflite"
    LABELS_PATH = "models/imageLabels.txt"

    interpreter = tf.lite.Interpreter(
      model_path = MODEL_PATH)
    interpreter.allocate_tensors()

    labels = load_labels(LABELS_PATH)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"INPUT {input_details}")
    print(f"OUTPUT {output_details}")

    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    input_mean = 127.5
    input_std = 127.5

    print("Running server on http://127.0.0.1:5000/")
    app.run(port=5000, debug=True)
