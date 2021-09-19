import cv2
import tensorflow as tf
import os
from keras import models

# img_array = cv2.normalize(img_array, img_array, alpha=5, beta=255, norm_type=cv2.NORM_MINMAX)  #stretching the histogram

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # img_array = cv2.bitwise_not(img_array)  #Needs to be uncommented on processed
    # img_array = cv2.resize(img_array, (28, 28)) #Needs to be uncommented on processed
    test = img_array.reshape(1,28,28,1)
    return test

model = tf.keras.models.load_model("mnist_az.h5")

print(model.summary())

prediction_list = []
file_names = []

directory = 'C:/Users/Dawid/PycharmProjects/NeuralNetwork/test_data_processed'
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        data = os.path.join(directory, filename)
        prediction = model.predict(prepare(data))
        prediction_list.append(prediction)
        file_names.append(filename)

#print(prediction)

data = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

iterator = 0

for j in range(len(prediction_list)):
    for i in range(35):
        if prediction_list[j][0][i] == 1:

            if file_names[j][0] == data[i]:
                state = "correct :)"
                iterator = iterator + 1
            else:
                state = "WRONG!"

            print(file_names[j], ": ", data[i]," ",state)

            break
print("Test correctness: ",round(((iterator/len(file_names))*100),2),"%")



