import os
import io
import cv2
import base64
import time
import json
import pickle
import numpy as np
import configparser

from os.path import join, abspath, dirname
from flask import Flask, request, jsonify

from main import loadModel, features_extraction


config = configparser.ConfigParser()
config.read(join(
                join(
                    dirname(abspath(__file__)), 'utils')
                , 'config.ini')
            )

app = Flask(__name__)

base_path = dirname(abspath(__file__))
model_path = join(base_path, 'models')
if config['model']['architecture'] == 'resnet50v2':
    model_architecture_path = join(model_path, 'resnet50v2')
    feature_list_path = join(model_architecture_path, config['model']['feature_list'])
    filenames_path = join(model_architecture_path, config['model']['filenames'])
    trigger_model = loadModel()
    model, feature_list, filenames, preprocess_input = trigger_model.begin(feature_list_path, filenames_path, config['model']['architecture'])
else:
    model_architecture_path = join(model_path, 'vgg19')
    feature_list_path = join(model_architecture_path, config['model']['feature_list'])
    filenames_path = join(model_architecture_path, config['model']['filenames'])
    trigger_model = loadModel()
    model, feature_list, filenames, preprocess_input = trigger_model.begin(feature_list_path, filenames_path, config['model']['architecture'])


@app.route('/api/image-search-tokoquick', methods=["POST", "GET"])
def predict():
    if request.method != "POST":
        return "API Image Search Tokoquick"
    
    start_time = time.time()
    image_file = request.files["image"]
    image_bytes = image_file.read()
    img_b64 = base64.b64encode(image_bytes)
    nparr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
    main = features_extraction()
    preds = main.get_prediction(image, model, feature_list, filenames, preprocess_input)
    temp_json = io.StringIO()
    preds.to_json(temp_json, orient='records')
    temp_json.seek(0)
    data_json = json.load(temp_json)
    result = {
        "success": True,
        "preds": data_json,
        "time":round(time.time() - start_time, 2)
    }
    result_json = jsonify(result)
    return result_json

if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0")