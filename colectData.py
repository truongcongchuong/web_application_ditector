import mediapipe as mp
import cv2 
import numpy as np
import pandas as pd

mp_Hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand = mp_Hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
cap = cv2.VideoCapture(0)
Label = input("nhập hành động thêm vào:")
Label = Label.lower()
frame_per_act = 40
count = 0
status = True
PATH_DATA = "DATA/DataTrain.csv"
PATH_LABEL = "DATA/label.csv"
DATA = []
LABEL = []
while cap.isOpened():
    ret, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resulft = hand.process(imgRGB)
    multi_landmarks = resulft.multi_hand_landmarks
    if multi_landmarks:
        num_hand = len(multi_landmarks)
        print(num_hand)
        hand_list = []
        for hand_lm in multi_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lm, mp_Hands.HAND_CONNECTIONS)
            for idx,lm in enumerate(hand_lm.landmark):
                hand_list.append(lm.x)
                hand_list.append(lm.y)
                hand_list.append(lm.z)
        if num_hand == 1:
            no_hand = np.zeros((len(hand_list)))
            no_hand = list(no_hand)
            hand_list = hand_list + no_hand
        DATA.append(hand_list)
        if count == frame_per_act - 1:
            count = 0
            LABEL.append(Label)
            if status == False:
                break
        else:
            count += 1
            
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) == ord("q"):
        status = False
cap.release()
cv2.destroyAllWindows()

DataFrame = pd.DataFrame(DATA)
LabelFrame = pd.DataFrame(LABEL)
print(DataFrame)

DataFrame.to_csv(PATH_DATA, columns=None, index=None, mode="a", header=False)
LabelFrame.to_csv(PATH_LABEL, columns=None, index=None, mode="a", header=False)