import random
import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


def generate_data(num_samples):
    df = pd.read_csv('../data_files/dataset3.csv')
    data_array = df.values
    return  data_array

def generate_data_periodically(interval):
    while True:
        data_array = generate_data(1)
        # print(data_array)
        time.sleep(interval)
def predict(data_array):
    X = data_array[:, :-1]
    Y = data_array[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = joblib.load('/home/viwichi/Projects/PycharmProjects/classifier_001/trained_model/model_001.joblib')
    prediction = clf.predict(X_test)

    return  prediction

def train(data_array):
    X = data_array[:, :-1]
    y = data_array[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, predictions))
    joblib.dump(clf, '../trained_model/model_001.joblib')
    print("\nModel saved successfully.")

def firebase_connect():
    cred = credentials.Certificate("/home/viwichi/Projects/PycharmProjects/classifier_001/credentials/postfix-ce6e0-firebase-adminsdk-q94qi-2147d94e6f.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://postfix-ce6e0-default-rtdb.firebaseio.com/'
    })
    ref = db.reference('/')
    return ref


def firebase_push(connected, posture):
    connected.set({
        "Posture":
            {
                "current": posture
            }
    })
    print(connected.get())

connected = firebase_connect()
while True:
    data_array = generate_data(1)
    predicted_arr = predict(data_array)
    for i in predicted_arr:
        print(i)
        firebase_push(connected, i)
        time.sleep(1)
    time.sleep(1)