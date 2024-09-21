import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM,Dense # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Dataset')

# Actions that we try to detect
actions = np.array(['hello', 'good'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequences_length = 30

label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequences_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        
X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir) 

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662))) # Match input shape
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))  # Number of actions

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
#! Change (epochs) as per the need of the datasets
model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])
model.summary()

res = model.predict(X_test)
print(actions[np.argmax(res[1])])
print(actions[np.argmax(y_test[1])])

model.save('model.keras')