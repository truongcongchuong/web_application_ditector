from flask import Flask, request, render_template
import requests as rq
from moviepy.editor import VideoFileClip
import cv2
from werkzeug.utils import secure_filename
import mediapipe as mp
import h5py
from keras.models import load_model
file = h5py.File("model.h5",mode="r")
app = Flask(__name__)
@app.route('/')
def index():
    return render_template("web_ditector.html")
@app.route('/frame', methods=["GET","POST"])
def handleVideo():
    try:
        video = request.files['video']
        filename = secure_filename(video.filename)
        print(filename)
        video.save(filename)
        cap = cv2.VideoCapture(f"{filename}")
        data = []
        while cap.isOpened():
            ret, frame = cap.read()
            print(frame)
            data.append(frame)
            cv2.imshow("image", frame)
            if cv2.waitKey(1):
                break
        cap.release()
        cv2.destroyAllWindows()
        return "phân tích hình ảnh"
    except Exception as e:
        return e

if __name__ == "__main__":
    app.run(debug=True)