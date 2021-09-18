import cv2
from flask import Flask, request

from services.textDetectorService import text_detector

app = Flask(__name__)


@app.route('/')
def index():
    return 'Working'


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    img = cv2.imread(uploaded_file.filename)

    imageO = cv2.resize(img, (640, 320), interpolation=cv2.INTER_AREA)
    textDetected = text_detector(imageO)

    print(textDetected)
    return textDetected


print(__name__)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
