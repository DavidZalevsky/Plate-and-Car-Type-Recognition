from analitic_plate_cut_jupyter import*
from pytesseract_plate_to_char_processing import*
from character_image_neural_processing import*
from ml_transfer_learning import*
from sql_data import*

#------------INPUT------------
input_1_plate_view = 'input_plate_view.png'
input_2_general_photo = 'car.jpg'

#------------FUNCTIONS------------

#Cutting plate from whole input image with analitic algorithm, saving to file "cropped_plate_image.png"
analitic_plate_cut(input_1_plate_view)

#Loading "cropped_plate_image.png" and processing to characters image with pytesseract, saving to folder /character from plate
plate_to_char_processing()

#Finding plate string with own trained neural network (conv)
plate_string = neural_processing()

#Loading input image of general car photo, to match car type with transfer learning
car_type = car_type_recognition(input_2_general_photo)

#Managing sql data with plate string and type of car. Every car that tried to entry garage is saved on google sql firebase.
# Also sending email if it's on blacklist.

sql_data = [plate_string,car_type]  #Example: sql_data = ['DW11111','garbage_truck']
sql_operation(sql_data)

