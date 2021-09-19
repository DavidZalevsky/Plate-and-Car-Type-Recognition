import cv2
import pytesseract
import os

def plate_to_char_processing():
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' #pytesseract initialize
    filename = 'cropped_plate_image.png'
    save_path = 'C:/Users/Dawid/PycharmProjects/NeuralNetwork/character_from_plate'
    iterator = 0

    # read the image and get the dimensions
    img = cv2.imread(filename)

    boxes = pytesseract.image_to_boxes(img) # also include any config options you use

    data = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","R","S","T","U","V","W","X","Y","Z"]

    # draw the bounding boxes on the image
    for b in boxes.splitlines():
        if b[0] in data:
            iterator = iterator + 1
            b = b.split(' ')

            y1 = int(b[2])
            y2 = int(b[4])
            x1 = int(b[1])
            x2 = int(b[3])

            crop_img = img[ y1:y2 , x1:x2 ]
            crop_img_processed = image_processing(crop_img)
            cv2.imwrite(os.path.join(save_path, str(iterator) + '.png'), crop_img_processed)

def image_processing(crop_img):
    white = [255, 255, 255]
    shape = crop_img.shape
    y_max = shape[0]
    x_max = shape[1]
    border = int((y_max - x_max) / 2)
    try:
        image = cv2.copyMakeBorder(crop_img,0,0,border,border,cv2.BORDER_CONSTANT,value=white) #Adding border
    except:
        image = cv2.normalize(crop_img, crop_img, alpha=5, beta=255, norm_type=cv2.NORM_MINMAX) #stretching the histogram
        image = cv2.resize(image, (28, 28))  # Resizing
        return (image)

    image = cv2.normalize(image, image, alpha=5, beta=255, norm_type=cv2.NORM_MINMAX) #stretching the histogram
    image = cv2.resize(image, (28, 28)) #Resizing
    return(image)

plate_to_char_processing()