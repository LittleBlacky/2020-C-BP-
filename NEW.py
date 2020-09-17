import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import csv
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist
modelfile = 'modelweight.model'
A=[1, 2, 6, 7, 8, 9, 13, 15, 17, 18, 21 , 28,25,31, 37, 48, 51, 54, 59, 64, 71, 80, 81, 91]
B=[3, 4, 5, 10, 12, 20,16, 26, 24, 32, 35,29, 38, 42, 43, 44, 55, 57, 58, 61, 63, 65, 70, 79, 84, 89]
C=[49, 67, 68, 73, 74, 76, 82, 88, 95, 102, 103, 107, 108, 109, 111, 114, 119, 121, 122, 123]
D=[11,22, 33,39, 41, 45, 50, 52, 53, 56, 60, 62, 66, 69, 72, 77, 78, 85, 87, 90, 93, 98, 99, 100, 113]
E=[19,14, 23,27,30, 36,34, 40, 46, 47, 75, 83, 86, 92, 94, 96, 97, 101, 104, 105, 106, 110, 112, 115, 116, 117, 118, 120]
Label=[]
for i in range(123):
    print(i)
    if i+1 in A:
        Label.append(0)
    if i+1 in B:
        Label.append(1)
    if i+1 in C:
        Label.append(2)
    if i+1 in D:
        Label.append(3)
    if i+1 in E:
        Label.append(4)

y_train= np.zeros((85,5),dtype = np.int)
for i in range(85):
    y_train[i][Label[i]]=1
y_test= np.zeros((38,5),dtype = np.int)
for i in range(38):
    y_test[i][Label[i]]=1

train=[]
test=[]
with open('TRA.csv', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for j in reader:
        train.append(list(map(float,j)))
    AA=(np.array(train))
    print(np.shape(AA))

with open('TES.csv', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for j in reader:
        test.append(list(map(float,j)))
    BB=np.array(test)
    print(np.shape(BB))

model=Sequential()
model.add(Dense(input_dim=4,units=100,activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(units=5,activation='softmax'))

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(AA,y_train,batch_size=123,epochs=500)
weights = np.array(model.get_weights())
result=model.evaluate(BB,y_test)
result2=model.evaluate(AA,y_train)
pre=model.predict(BB)


AC=[]
BC=[]
np.savetxt("data.txt", pre)





#model.save_weights(modelfile) #保存模型权重

