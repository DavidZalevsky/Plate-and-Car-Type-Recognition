import cv2
import os

def image_processing(img):
    white = [255,255,255]
    shape = img.shape
    y_max = shape[0]
    x_max = shape[1]
    border = int((y_max - x_max)/2)
    try:
        image_with_border = cv2.copyMakeBorder(img,0,0,border,border,cv2.BORDER_CONSTANT,value=white) #Adding border
    except:
        image_with_border = cv2.bitwise_not(img)  # Negative of image
        image_with_border = cv2.resize(image_with_border, (28, 28))  # Resizing
        return (image_with_border)

    image_with_border = cv2.bitwise_not(image_with_border) #Negative of image
    image_with_border = cv2.resize(image_with_border, (28, 28)) #Resizing
    return(image_with_border)

save_home_dir = "C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_az_data_full_machine_pre_aug"
load_home_dir = "C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_az_data_full_machine_raw"

iterator = 0
#For each folder
for load_home_dir, folders_home, files_home in os.walk(load_home_dir):
   for folder in folders_home:
      path_load = (os.path.join(load_home_dir, folder))

      #For each file in folder
      for path_load, folders, files in os.walk(path_load):
         for file in files:
            iterator = iterator + 1
            path_load_final = (os.path.join(path_load, file))

            path_save = os.path.join(folder,file)
            path_save_final = os.path.join(save_home_dir,path_save)

            image = cv2.imread(path_load_final)

            image_processed = image_processing(image)

            cv2.imwrite(path_save_final,image_processed)