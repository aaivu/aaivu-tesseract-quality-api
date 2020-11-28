import io
from base64 import encodebytes

import cv2
import quality_checker
from PIL import Image
from flask_api.error import Error
from flask import Flask, request, jsonify


# from flask import jsonify

def get_response_image(image_path):
  pil_img = Image.open(image_path, mode='r')  # reads the PIL image
  byte_arr = io.BytesIO()
  pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
  encoded_img = encodebytes(byte_arr.getvalue()).decode(
      'ascii')  # encode as base64
  return encoded_img


UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app = Flask(__name__)


@app.route("/test")
def hello():
  return "Hello World!"
  l


@app.route("/oculars/inputImage/quality", methods=["POST"])
def process_image():
  file = request.files['image']
  image = Image.open(file)
  image.save('temp.png')
  response, yolo_img = quality_checker.run(image_path='temp.png')
  if type(response) == Error:
    return response.response, response.status
  else:
    image_path = "apk/static/output.png"
    cv2.imwrite(image_path, yolo_img)
    enocded_image = get_response_image(image_path)
    return jsonify({'Status': 'Success', "ImageBytes": enocded_image})


if __name__ == "__main__":
  app.run(debug=True, port=5000, host="0.0.0.0")
