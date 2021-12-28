import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt


def check(element):
  try:
    float(element)
    return True
  except ValueError:
    return False


def predictor(model_path, data_path):
    model = load_model(model_path)
    data = pd.read_csv(data_path).columns.to_numpy()

    indexs = []
    elements = []
    data = list(data)
    for ele in data:
        if check(ele) is False:
            indexs.append(data.index(ele))
            elements.append(ele)

    #preprocessing data because it's string data and append '.1' at the end of datapoint so we must cut it
    for i in indexs:
        data[i] = float(data[i][:-2])

    for idx in range(len(data)):
        data[idx] = float(data[idx])

    #reshape input to fit with model
    data = np.array(data)
    # new_data = data.reshape(1, 1, 3000) #lstm
    new_data = data.reshape(1, 3000, 1)  # transformer, cnn

    result = model.predict(new_data)
    label_predict = np.argmax(result)

    return result, label_predict, data


def return_label(label_pred):
    if label_pred == 0:
        return "bad"
    elif label_pred == 1:
        return "good"


if __name__ == '__main__':
    # labels = []
    for i in range(34):
        data_path = './Data_test/data_' + str(i+1) +'.csv'
        model_path = './Model/fully_CNN_model.h5'
        result, label_predict, data = predictor(model_path, data_path)
        # print(f'Result: {result}')
    #     labels.append(return_label(label_predict))
    # print(labels)
        # print(f'This is a {return_label(label_predict)} PPG signal.')
        title = return_label(label_predict) + " data"
        plt.figure()
        plt.plot(data)
        plt.title(title)
        plt.savefig('./Result/fully_CNN/fully_CNN'+ 'Data_'+str(i+1)+ '.jpg')
        # plt.show()

