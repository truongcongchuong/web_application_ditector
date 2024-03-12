import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score
import numpy as np

# đọc dữ liệu 
path_data = "DATA/"
path_train = path_data + "DataTrain.csv"
path_label = path_data + "label.csv"
df_feature = pd.read_csv(path_train, header=None)
df_label = pd.read_csv(path_label, header=None)
label = df_label.values

df_feature["group"] = df_feature.index //40
datasets = {g:v for g,v in df_feature.groupby("group")}

X = np.stack([datasets[i].drop('group', axis=1).values for i in datasets.keys()])

label = label.ravel()

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(label)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_word = len(label_encoder.classes_)

# tạo mô hình LSTM

model = Sequential()
model.add(LSTM(32, return_sequences=True, activation='relu',input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_word, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.summary()
model.save("model.h5")

predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
print(predictions)
print(label_encoder.inverse_transform(predicted_classes))

y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, predicted_classes)
print(f"độ chính xác:{accuracy*100}%")