import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

accuracy_data = []
data = [1,2,3,4,5,6,7,8,9,10]
for cv in data:

    model = keras.models.load_model('MC_3DCNN\model\model_epoch_500_'+str(cv)+'.h5') # 모델 파일 경로를 지정하세요


        # 모델 로드


    # 입력 데이터 준비 (예시)
    input_data = np.load('MC_3DCNN\split_data_10/train_data_list_'+str(cv)+'.npy')  # your_input_data에 입력 데이터를 넣으세요
    label = np.load('MC_3DCNN\split_data_10/train_label_list_'+str(cv)+'.npy')

    predicted_labels = []
    # print(input_data)
    for y, i in enumerate(input_data):
        i = i.reshape(1,192,192,32)
    #    # print(i.shape) 
        a = model.predict(i)
        #print(a)
        a_max = 0
        max = 0
        if a[0][0] > max:
            max = a[0][0]
            a_max = 1
        if a[0][1] > max:
            max = a[0][1]
            a_max = 2
        if a[0][2] > max:
            max = a[0][2]
            a_max = 3
        print(a_max, label[y])
        predicted_labels.append(a_max)
        
    print(predicted_labels)
    accuracy = accuracy_score(label, predicted_labels)
    b = accuracy * 100
    b = round(b, 2)
    accuracy_data.append(b)
print(accuracy_data)


np.save("all_accuracy_data.npy",accuracy_data)


