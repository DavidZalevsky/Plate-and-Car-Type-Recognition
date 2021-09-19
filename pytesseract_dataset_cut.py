import cv2
import pytesseract
import os
import matplotlib.pyplot as plt

#------------Capturing data with pytesseract------------
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' #pytesseract initialize
alphabet_dataset_dir = 'C:/Users/Dawid/PycharmProjects/NeuralNetwork/alphabet_dataset'

image_iterator = 0
totalFiles = 0
totalDir = 0
for base, dirs, files in os.walk(alphabet_dataset_dir):
    print('Searching in : ',base)
    for directories in dirs:
        totalDir += 1
    for Files in files:
        totalFiles += 1

for i in range(totalFiles): #for every alphabet image in dir

    image_path = os.path.join(alphabet_dataset_dir,files[i])

    cropped_image = cv2.imread(image_path) #loading cropped plate
    hImage,wImage,_ = cropped_image.shape #loading plate size
    letters_data_list = pytesseract.image_to_boxes(cropped_image) #letter recognition with pytesseract

    letters_data_list_temp = letters_data_list.splitlines() #initialy parsed data
    letters_data_list_parsed = [] #final parsed plate data list
    for letter_number in range(len(letters_data_list_temp)):
        letters_data_list_parsed.append(letters_data_list_temp[letter_number].split())
        #img = cv2.rectangle(img, (int(letters_data_list_temp[letter_number][1]), hImage - int(letters_data_list_temp[letter_number][2])), (int(letters_data_list_temp[letter_number][3]), hImage - int(letters_data_list_temp[letter_number][4])), (0, 255, 0), 2)
        #letters_data_list_parsed[x][y] #x means certain letter data info, y means certain data from certain letter (data format: letter_name|x1|y1|x2|y2|0)

    for letter_number in range(len(letters_data_list_parsed)):
        break


    #------------Saving to folder------------

    data = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","R","S","T","U","V","W","X","Y","Z"]

    base_dir = 'C:/Users/Dawid/PycharmProjects/NeuralNetwork/minst_az_data_full_machine_raw'

    for letter in range(len(letters_data_list_parsed)): #len(letters_data_list_parsed)

        image_iterator = image_iterator + 1

        print("Progress: ",(round(image_iterator/1435,3)*100),"%")

        folder_path = os.path.join(base_dir,letters_data_list_parsed[letter][0]) #path save
        final_path = os.path.join(folder_path,(str(image_iterator) + ".png"))

        t1 = (int(letters_data_list_parsed[letter][1]))
        t2 = (hImage - int(letters_data_list_parsed[letter][2]))
        t3 = (int(letters_data_list_parsed[letter][3]))
        t4 = (hImage - int(letters_data_list_parsed[letter][4]))
        offset = 3

        crop_letter = cropped_image[t4-offset:t2+offset,t1-offset:t3+offset]  # 2,4,1,3 #cutting certain letter
        #crop_letter = cv2.resize(crop_letter, (28, 28)) #image resize
        crop_letter = cv2.cvtColor(crop_letter, cv2.COLOR_BGR2GRAY) #trasnforming to gray
        #crop_letter = cv2.bitwise_not(crop_letter) #negative (to match mnist dataset)
        cv2.imwrite(final_path, crop_letter)
