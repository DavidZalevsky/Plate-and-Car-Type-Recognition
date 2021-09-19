import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' #pytesseract initialize

filename = 'alphabet.png'

# read the image and get the dimensions
img = cv2.imread(filename)
h, w, _ = img.shape # assumes color image

# run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_boxes(img) # also include any config options you use

data = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","R","S","T","U","V","W","X","Y","Z"]
letter_list = []

# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    letter_list.append(b[0])

print(letter_list,"Letter list")

iterator = 0
for i in range(len(data)):
    try:
        if letter_list[i] in data:
            data.remove(letter_list[i])
    except:
        print("ERROR")
        break

if data == []:
    if len(letter_list) == 35:
        print("List complete")
    else:
        print("List has bad boxes")
else:
    print("List not complete: ",data)

print(letter_list)

# show annotated image and wait for keypress
cv2.imshow(filename, img)
cv2.waitKey(0)