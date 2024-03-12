from flask import Flask, request, render_template
import cv2
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp
import h5py
import numpy as np
from keras.models import load_model
import pandas as pd

file = h5py.File("model.h5",mode="r")
model = load_model(file)

mp_Hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand = mp_Hands.Hands()

label = pd.read_csv("DATA/label.csv").values
label = label.ravel()
label_encoder = LabelEncoder()
actions = label_encoder.fit_transform(label)
frame_per_act = 40

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
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resulft = hand.process(imgRGB)
                multi_landmarks = resulft.multi_hand_landmark
                if multi_landmarks:
                    num_hand = len(multi_landmarks)
                    print(num_hand)
                    hand_list = []
                    for hand_lm in multi_landmarks:
                        for idx,lm in enumerate(hand_lm.landmark):
                            hand_list.append(lm.x)
                            hand_list.append(lm.y)
                            hand_list.append(lm.z)
                    if num_hand == 1:
                        no_hand = np.zeros((len(hand_list)))
                        no_hand = list(no_hand)
                        hand_list = hand_list + no_hand
                    data.append(hand_list)
                os.remove(filename)
                cv2.imshow("image", frame)  
                cv2.waitKey(1)
                   
            cap.release()
            cv2.destroyAllWindows()
        except:
        # xử lý dữ liệu đầu vào và dự đoán
            df = pd.DataFrame(data, columns=None, index=None).values
            new_frame = []
            test_frame = []
            for fr in df:
                if len(test_frame) == frame_per_act:
                    new_frame.append(test_frame)
                    test_frame = []
                test_frame.append(fr)
            if len(test_frame) < frame_per_act:
                missing_data = frame_per_act - len(test_frame)
                null_frame = np.ones((missing_data, 126))
                null_frame[:] = None
                new_frame = np.concatenate((np.array(df), null_frame), axis=0)
            df = pd.DataFrame(new_frame, columns=None, index=None)
            df = df.fillna(method="pad")
            df["group"] = df.index // frame_per_act
            datasets = {g:v for g,v in df.groupby("group")}
            DataPredict = np.stack([datasets[i].drop('group', axis=1).values for i in datasets.keys()])
            print(len(DataPredict))
            
            predictions = model.predict(DataPredict)
            predicted_classes = np.argmax(predictions, axis=1)
            predictions_resulft = label_encoder.inverse_transform(predicted_classes)
            centence = " ".join(predictions_resulft)
            return centence
    except Exception as e:
        return e

if __name__ == "__main__":
    app.run(debug=True)