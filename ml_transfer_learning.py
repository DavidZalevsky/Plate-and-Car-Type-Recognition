#from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
#from tensorflow.keras.applications.resnet152 import ResNet152
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

def car_type_recognition(img):
    model = ResNet101V2(weights='imagenet') #working on ResNet50
    img = image.load_img(img, target_size=(224, 224)) #Loading image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    type_of_car = decode_predictions(preds,top=1)[0][0][1] #Type of vehicle saved as string

    return(type_of_car)