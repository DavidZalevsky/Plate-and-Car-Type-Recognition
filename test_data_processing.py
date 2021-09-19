import cv2
import os

load_home_dir = 'C:/Users/Dawid/PycharmProjects/NeuralNetwork/test_data'
save_home_dir = 'C:/Users/Dawid/PycharmProjects/NeuralNetwork/test_data_processed'

def image_processing(image):
   ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
   image = cv2.bitwise_not(image)
   image = cv2.resize(image, (28, 28))
   return(image)

#For each folder
for load_home_dir, folders_home, files_home in os.walk(load_home_dir):
   for files in files_home:
      path_load = os.path.join(load_home_dir, files)
      path_save = os.path.join(save_home_dir,files)

      image = cv2.imread(path_load, cv2.IMREAD_GRAYSCALE)
      image = image_processing(image)

      cv2.imwrite(path_save,image)