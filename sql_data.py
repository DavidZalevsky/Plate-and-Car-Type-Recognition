import pyrebase
import firebase_admin
from admin_config import *
from firebase_admin import credentials, firestore
import smtplib, ssl
from gui import *

#------------- INITIALIZING FIREBASE-------------

def initialize():
    global firestore_db,firebase,auth

    firebaseConfig = {
        'apiKey': "AIzaSyA24hg_POVw-BCYFCb7NZ-mhHxDR8SLvq0",
        'authDomain': "aurora-albion-fishbot.firebaseapp.com",
        'databaseURL': "https://aurora-albion-fishbot.firebaseio.com",
        'projectId': "aurora-albion-fishbot",
        'storageBucket': "aurora-albion-fishbot.appspot.com",
        'messagingSenderId': "559503484485",
        'appId': "1:559503484485:web:c1623ff05d59e7bd12e52c",
        'measurementId': "G-5C6719HYQ2"
    }

    #auth login
    firebase = pyrebase.initialize_app(firebaseConfig)
    auth = firebase.auth()

    # initialize sdk
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    # initialize firestore instance
    firestore_db = firestore.client()

def email(plate):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "adezalew@gmail.com"  # Enter your address
    receiver_email = "dezalew@gmail.com"  # Enter receiver address
    password = 'dawiddawid'
    message = "Subject: Car Alert\n\n"

    info = "Ten samochod z czarnej listy probowal wjechac na parking: "
    message = message + info + plate

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

def sql_operation(sql_data):

    initialize()  # initialize whole firebase
    info = True #default value about info of car
    visit_count = 0
    time_arrival = 0
    last_visit_old = 0
    visits_array = []

    plate = sql_data[0]
    type = sql_data[1]
    seconds = time.time() #time of car arrival

    firestore_db.collection(u'users').add({
        'plate': plate,
        'type': type,
        'time': seconds})

    if type == forbidden_vehicle:
        print('Niedozwolony typ auta! Brak dostępu do parkingu!')
        info = False

    if plate in black_list:
        print('AUTO ISTNIEJE NA CZARNEJ LIŚCIE, ADMINISTRATOR ZOSTAŁ POWIADOMIONY EMAILEM!')
        email(plate)
        info = False

    if info == True:
        print("Witamy na parkingu!")

        snapshots = list(firestore_db.collection(u'users').get())
        for snapshot in snapshots:
            user_data = (snapshot.to_dict())

            if user_data["plate"] == plate:
                visit_count = visit_count + 1
                last_visit = user_data["time"]
                if last_visit > last_visit_old:
                    visits_array.append(last_visit)

        if len(visits_array) == 1:
            time_arrival = visits_array[0]
        else:
            visits_array = sorted(visits_array)
            time_arrival = visits_array[-2]

        print("To twoja: ",visit_count," wizyta na tym parkingu!")
        print("Ostatnia wizyta odbyła się:", time.ctime(time_arrival))

    gui(info, visit_count, time_arrival, plate)

# sql_data = ['DW11112','car']
# sql_operation(sql_data)    ###