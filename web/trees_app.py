from flask import Flask, render_template, request, jsonify, Response
import io
from PIL import Image
from model import ImageClassifier
#import numpy as np

app = Flask(__name__)
classifier = ImageClassifier()

@app.route('/')
def index():
    return render_template('identifier.html')

@app.route('/identify', methods=['POST'])
def identify():
    file_object = request.files['image']
    image = file_object.read()
    print(type(image))
    predictions = image
    # img_bytes = io.BytesIO(image)
    # predictions = classifier.predict(image)
    return Response(response=predictions, status=200, mimetype="text/plain")

if __name__ == '__main__':
    #threaded=True creates separate thread for each person who visits website
    #can set debug=True, but turn it OFF before deploying in production
    app.run(host='0.0.0.0', port='5002', threaded=True)
