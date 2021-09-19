from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import random
import cv2
from tensorflow.keras import layers
import numpy as np
import os
from keras import models
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import SGD


iterator = 0

def data_loading():
    #Global variables
    global data,labels
    global iterator

    #Enter the path of your image data folder
    image_data_folder_path = "C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_az_data_full_machine_aug normal"

    # initialize the data and labels as an empty list
    #we will reshape the image data and append it in the list-data
    #we will encode the image labels and append it in the list-labels
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(image_data_folder_path)))

    #total number images
    total_number_of_images = len(imagePaths)
    print("\n")
    print("Total number of images----->",total_number_of_images)

    #randomly shuffle all the image file name
    random.shuffle(imagePaths)

    print("Data processing...")
    # loop over the shuffled input images
    for imagePath in imagePaths:

        #Read the image into a numpy array using opencv
        #all the read images are of different shapes
        image = cv2.imread(imagePath)

        #resize the image to be 32x32 pixels (ignoring aspect ratio)
        #After reshape size of all the images will become 32x32x3
        #Total number of pixels in every image = 32x32x3=3072

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #MINST_AZ

        # image = np.array(image).ravel()

        image = cv2.resize(image, (28,28))

        #image = image.reshape(len(image), image[0], image[1], image[2])

        #Append each image data 1D array to the data list
        data.append(image)

        print(iterator,"/",total_number_of_images," ",round((iterator*100/total_number_of_images),2),"%")
        iterator = iterator + 1

        # extract the class label from the image path and update the
        label = imagePath.split(os.path.sep)[-2]

        #Append each image label to the labels list
        labels.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    #convert the data and label list to numpy array
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    print("Data processing finished")
def data_split():
    global trainX, testX, trainY, testY, data



    print("Data splitting...")

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    # train_test_split is a scikit-learn's function which helps us to split train and test images kept in the same folders
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, shuffle=True)

    print("Number of training images--->", len(trainX), ",", "Number of training labels--->", len(trainY))
    print("Number of testing images--->", len(testX), ",", "Number of testing labels--->", len(testY))

    # convert the labels from integers to vectors
    # perform One hot encoding of all the labels using scikit-learn's function LabelBinarizer
    # LabelBinarizer fit_transform finds all the labels

    lb = preprocessing.LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    #trainY = to_categorical(trainY)
    #testY = to_categorical(testY)

    print("\n")
    print("Classes found to train", )
    train_classes = lb.classes_
    print(train_classes)
    binary_rep_each_class = lb.transform(train_classes)
    print("Binary representation of each class")
    print(binary_rep_each_class)
    print("\n")
def plot():
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["accuracy"], label="Test Accuracy")
    plt.plot(N, H.history["val_accuracy"], label="Validation Accuracy")
    plt.plot(N, H.history["loss"], label="Loss")
    plt.title("Wykres uczenia oraz walidacji")
    plt.xlabel("Epoka")
    plt.ylabel("Dokladnosc [%]")
    plt.legend()
    plt.show()

data_loading()
data_split()

EPOCHS = 15

trainX = trainX.reshape(len(trainX),28,28,-1)
testX = testX.reshape(len(testX),28,28,-1)

# trainX = np.expand_dims(trainX, -1)
# trainX.shape = (28,28,1)
#
# testX = np.expand_dims(testX, -1)
# testX.shape = (28,28,1)

#-----------------------ARCHITECUTRES-----------------------

#Data architecture dense
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(layers.Dense(35, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=500)

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.3))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.4))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(35, activation='softmax'))


# model = models.Sequential()
# model.add(layers.Dense(100, activation = 'relu', input_shape=(28*28,)))
# model.add(layers.Dense(50, activation = 'relu'))
# model.add(layers.Dense(35, activation='softmax'))
# model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=100)

#WINNER
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(35, activation='softmax'))
# opt = SGD(lr=0.001,momentum = 0.9)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=500)


#Data architecture dense

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(35, activation='softmax'))

opt = SGD(lr=0.01,momentum = 0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=100)

plot()
model.save('mnist_az.h5')