from keras.datasets import mnist
from PIL import Image
import os

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

folder_directorys = []

folder_directorys.append('C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_data/0')
folder_directorys.append('C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_data/1')
folder_directorys.append('C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_data/2')
folder_directorys.append('C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_data/3')
folder_directorys.append('C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_data/4')
folder_directorys.append('C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_data/5')
folder_directorys.append('C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_data/6')
folder_directorys.append('C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_data/7')
folder_directorys.append('C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_data/8')
folder_directorys.append('C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_data/9')

for i in range(len(train_images)):

    # data from label
    number = train_labels[i]

    # path to save depends of label
    path = os.path.join(folder_directorys[number],(str(i)+".png"))

    # Saving array to image in certain path
    image = Image.fromarray(train_images[i])
    image.save(path, 'PNG')