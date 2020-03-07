import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import metrics
knn = cv2.ml.KNearest_create()

def start(sample_size=25) :
    train_data = generate_data(sample_size)
    labels = classify_label(train_data)
    power, nomal, short = binding_label(train_data, labels)
    print("Return true if training is successful :", knn.train(train_data, cv2.ml.ROW_SAMPLE, labels))
    return power, nomal, short

#'num_samples' is number of data points to create
#'num_features' means the number of features at each data point (in this case, x-y conrdination values)
def generate_data(num_samples, num_features = 2) :
    """randomly generates a number of data points"""    
    data_size = (num_samples, num_features)
    data = np.random.randint(0,40, size = data_size)
    return data.astype(np.float32)


def run(new_data, power, nomal, short):
    a = np.array([new_data])
    b = a.astype(np.float32)    
    ret, results, neighbor, dist = knn.findNearest(b, 3) # Second parameter means 'k'
    print("predicted label : ", results)
    return int(results[0][0])

#I determined the drowsiness-driving-risk-level based on the time which can prevent driving accident.
def classify_label(train_data):
    labels = []
    for data in train_data :
        if data[1] < data[0]-15 :
            labels.append(2)
        elif data[1] >= (data[0]/2 + 15) :
            labels.append(0)
        else :
            labels.append(1)
    return np.array(labels)

def binding_label(train_data, labels) :
    power = train_data[labels==0]
    nomal = train_data[labels==1]
    short = train_data[labels==2]
    return power, nomal, short
