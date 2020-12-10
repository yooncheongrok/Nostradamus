import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense , LSTM, Dropout

import numpy as np
import pandas as pd

import seaborn as sns
from DataProcessing.DataProcessing import *
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as pdr
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json




class StockDataset():
    ### X_frame, y_frames = 시퀀스 랭스 정하는 변수.
    def __init__(self, symbol, x_frames, y_frames, start, end):

        ### StockDataset('028050.KS', 10, 5, (2001,1,1),(2005,1,1))로 삼성전자 호출.

        self.symbol = symbol
        self.x_frames = x_frames  ## sequnce.
        self.y_frames = y_frames  ## 우리는 하루 예측이라서 항상 1

        self.start = datetime.datetime(*start)  ##날짜받기.
        self.end = datetime.datetime(*end)  ##날짜받기.

        self.data = pdr.DataReader(self.symbol, 'yahoo', self.start, self.end)  ##야후에서 panda로 데이터를 보내줌.
        print("number of MAN")
        print(self.data.isna().sum())

        ###ADj Close Drop out
        self.data = self.data.drop(['Adj Close'], axis=1)

    ## sotfmax로 학습하기위해 y값 원핫 인코딩,
    def For_one_hot_train(self, train_rate):
        if 0 < train_rate and train_rate < 1:
            self.data_train = self.data[: int(self.data.shape[0] * train_rate)].copy()
            ###스케일 넘파이로 바뀜 !
            Scaler = MinMaxScaler()
            self.data_train = Scaler.fit_transform(self.data_train)
            ###트레인으로 짜름.

            self.X_data_train = []
            self.y_data_train = []

            #### one_hot 트레이닝 셋으로 만들기.
            for i in range(self.x_frames, self.data_train.shape[0]):

                self.X_data_train.append(self.data_train[i - self.x_frames: i])  ## 예를들어 x데이터 60일치 짜르기.

                if self.data_train[i, 3] >= self.data_train[i - 1, 3]:  ## 3번쨰 칼럼은 종가를 뜻함. y데이터는 종가보다 올랏으면 1 떨어졌으면 0
                    self.y_data_train.append(1)
                else:
                    self.y_data_train.append(0)

            self.X_data_train = np.array(self.X_data_train)
            self.y_data_train = np.array(self.y_data_train)

            ## X_data_train dim = (traing_set, sequence(X_frames) , colummns(x의 요소 ))
            return self.X_data_train, self.y_data_train

    def For_one_hot_test(self, test_rate):
        if 0 < test_rate and test_rate < 1:

            self.data_test = self.data[int(self.data.shape[0] * (1 - test_rate)):].copy()
            ###스케일
            Scaler = MinMaxScaler()
            self.data_test = Scaler.fit_transform(self.data_test)
            ###테스트으로 짜름.

            self.X_data_test = []

            self.y_data_test = []

            #### one_hot 테스트 셋으로 만들기.
            for i in range(self.x_frames, self.data_test.shape[0]):

                self.X_data_test.append(self.data_test[i - self.x_frames: i])  ## 예를들어 x데이터 60일치 짜르기.
                ## for문에서 마지막 포함 안하고 자르기에서 포함안해서 크기에서 -2 만큼 뒤에.

                if self.data_test[i, 3] >= self.data_test[i - 1, 3]:  ## 3번쨰 칼럼은 종가를 뜻함. y데이터는 종가보다 올랏으면 1 떨어졌으면 0
                    self.y_data_test.append(1)
                else:
                    self.y_data_test.append(0)

            self.X_data_test = np.array(self.X_data_test)
            self.y_data_test = np.array(self.y_data_test)

            return self.X_data_test, self.y_data_test

    ###하루예측 dim 맞춰주기. (# of training, # of sequence, # of x의 요소 )
    def For_tommorow(self):
        Scaler = MinMaxScaler()
        self.data_for_tommorow = self.data[self.data.shape[0] - self.x_frames:self.data.shape[0]]
        self.data_for_tommorow = Scaler.fit_transform(self.data_for_tommorow)
        self.data_for_tommorow = np.array(self.data_for_tommorow).reshape(1, self.x_frames, -1)
        return self.data_for_tommorow


### 로지스틱 정확도 함수 (label 이 두개일 때 . )
def accuracy(y_pred, y_test):
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1

    result = 0
    for i in range(0, y_pred.shape[0]):
        if y_pred[i] == y_test[i]:
            result += 1

    accuracy = (result / y_pred.shape[0]) * 100

    return accuracy

datacode_dic = {'삼성전자': '005930.KS', 'SK하이닉스': '000660.KS', 'LG화학': '051910.KS',
                '삼성바이오로직스': '207940.KS','삼성전자우' : '005935.KS','셀트리온':'068270.KS',
                'NAVER':'035420.KS','현대차':'005380.KS','삼성SDI':'006400.KS',
                '카카오':'035720.KS', '기아차':'000270.KS','LG생활건강':'051900.KS',
                '현대모비스':'012330.KS','POSCO':'005490.KS','삼성물산':'028260.KS'}


dataset = StockDataset('028260.KS', 60, 1, (2001, 1, 1), (2020, 11, 21))  ##005930.KS ,028050.KS

X_train, y_train = dataset.For_one_hot_train(0.8)
X_test, y_test = dataset.For_one_hot_test(0.2)
X_for_tommorow = dataset.For_tommorow()

##모델.
regression = Sequential()
regression.add(LSTM(units=60, activation='relu', return_sequences=True,
                    input_shape=(X_train.shape[1], X_train.shape[2])))
regression.add(Dropout(0.1))

### 마지막 아웃풋 many to one 모델.
regression.add(LSTM(units=60, activation='relu'))  ###마지막 아웃풋이라 시퀀스 리턴이 필요가없음. 아웃풋 1나와야됨.
regression.add(Dropout(0.1))

### 마지막 AL을 다시 fully nn 에 연결. 로지스틱이므로 마지막은 시그모이드로 출력
regression.add(Dense(units=1))
regression.add(layers.Activation(activation='sigmoid'))

regression.summary()  ### [batche, 인풋, 아웃풋]

## amdam으로 초적화 하고 loss 함수는 label 이 2개임으로 binary_crossentorpy를 이용하였음


regression.compile(optimizer='adam', loss='binary_crossentropy')

### training 시작.

regression.fit(X_train, y_train, epochs=10, batch_size=32, shuffle=True)

regression.save("../Model/028260.h5")

##예측값.
y_pred_train = regression.predict(X_train)
y_pred_test = regression.predict(X_test)
y_pred_tommorow = regression.predict(X_for_tommorow)
up = y_pred_tommorow[0, 0]
down = 1 - y_pred_tommorow[0, 0]

# In[8]:


probability = {"probability": {"up": str(up), "down": str(down)}}

# In[9]:


##정확도 계산과 내일 몇 % 오를지 측정.
print("train accuracy : %s %%" % accuracy(y_pred_train, y_train))
print("test accuracy : %s %%" % accuracy(y_pred_test, y_test))
print("up : %s %%, down : %s %%" % (up * 100, down * 100))  ## 종목 내일의 확률

# In[10]:


probability_json = json.dumps(str(probability), indent=4)

# In[11]:


########### 데이터베이스에 올리기.

# cred = credentials.Certificate('stock-b69a2-firebase-adminsdk-t55a0-66534d7508.json')
# ##키 파일
# firebase_admin.initialize_app(cred, {'databaseURL': 'https://stock-b69a2.firebaseio.com'})
# dir = db.reference()
# ## 데이터베이스 접근.
#
# dir.update(probability)
# # In[13]:


# upload = Database(probability)