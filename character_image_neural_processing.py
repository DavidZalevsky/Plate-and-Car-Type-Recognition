import cv2
import tensorflow as tf
import os

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    #ret, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
    img_array = cv2.bitwise_not(img_array)
    img_array = cv2.resize(img_array, (28, 28)) #double check
    test = img_array.reshape(1,28,28,1)
    return test

def neural_processing():
    model = tf.keras.models.load_model("mnist_az.h5")
    prediction_list = []
    file_names = []

    directory = 'C:/Users/Dawid/PycharmProjects/NeuralNetwork/character_from_plate'
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            data = os.path.join(directory, filename)
            prediction = model.predict(prepare(data))
            prediction_list.append(prediction)
            file_names.append(filename)

    data = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
            "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    plate_characters = []

    for j in range(len(prediction_list)):
        for i in range(35):
            if prediction_list[j][0][i] == 1:
                plate_characters.append(data[i])

    plate_characters = ''.join(plate_characters)

    return(plate_characters)

#print(neural_processing())




