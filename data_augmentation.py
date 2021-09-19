from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os

def data_augmentation(img):
    # convert to numpy array
    data = img_to_array(img)

    # expand dimension to one sample
    samples = expand_dims(data, 0)

    # create image data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range= 30,
        width_shift_range= 0.05,
        height_shift_range= 0.05,
        shear_range= 0.1,
        zoom_range= 0.1,)

    # prepare iterator
    it = datagen.flow(samples, batch_size=1)

    # generate batch of images
    batch = it.next()

    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')

    return(image)

#load/save dir
load_home_dir = "C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_az_data_full_machine_pre_aug"
save_home_dir = "C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_az_data_full_machine_aug"

#Augmentation multiplyer, default is 150 * 41 = 6150
aug_multiplier = 65

iterator = 0
#For each folder
for load_home_dir, folders_home, files_home in os.walk(load_home_dir):
   for folder in folders_home:
      path_load = (os.path.join(load_home_dir, folder))

      #For each file in folder
      for path_load, folders, files in os.walk(path_load):
         for file in files:
            path_load_final = (os.path.join(path_load, file))

            # load the image
            image = load_img(path_load_final)

            for i in range(aug_multiplier):
                iterator = iterator + 1
                file_editet = str(i) + file
                path_save = os.path.join(folder, file_editet)
                path_save_final = os.path.join(save_home_dir, path_save)

                # image augomentation
                image_augmented = data_augmentation(image)

                # save image
                cv2.imwrite(path_save_final,image_augmented)

                print("Progress: ",round(iterator/(35*aug_multiplier*41),3)*100,"%")