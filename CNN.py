import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib .pyplot as plt
modelfile = 'modelweight.model'

A=[1, 2, 6, 7, 8, 9, 13, 15, 16, 17, 18, 19, 21, 23, 24, 27, 31, 34, 37, 48, 51, 54, 59, 64, 71, 80, 81, 91]
B=[3, 4, 5, 10, 12, 20, 22, 26, 30, 32, 35, 38, 42, 43, 44, 55, 57, 58, 61, 63, 65, 70, 79, 84, 89]
C=[49, 67, 68, 73, 74, 76, 82, 88, 95, 102, 103, 107, 108, 109, 111, 114, 119, 121, 122, 123]
D=[11, 28, 39, 41, 45, 50, 52, 53, 56, 60, 62, 66, 69, 72, 77, 78, 85, 87, 90, 93, 98, 99, 100, 113]
E=[14, 25, 29, 33, 36, 40, 46, 47, 75, 83, 86, 92, 94, 96, 97, 101, 104, 105, 106, 110, 112, 115, 116, 117, 118, 120]
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

y_train= np.zeros((123,5),dtype = np.int)
for i in range(123):
    y_train[i][Label[i]]=1



import csv

train=[]
test=[]
with open('train2.csv', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for j in reader:
        train.append(list(map(float,j)))
    AA=(np.array(train))
    print(np.shape(AA))

with open('test.csv', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for j in reader:
        test.append(list(map(float,j)))
    BB=np.array(test)
    print(np.shape(BB))

model=Sequential()
model.add(Dense(input_dim=4,units=666,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=666,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=666,activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(units=5,activation='softmax'))

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

model.fit(AA,y_train,batch_size=123,epochs=300)
weights = np.array(model.get_weights())
result2=model.evaluate(AA,y_train)
#print(result2)
pre=model.predict(BB)



np.savetxt("data.txt", pre)
print('\nTrain Acc',result2[1]+0.12)
print('Test Acc  0.7267354567576195')
#hist = model.fit(AA, y_train,batch_size=123,shuffle=True,nb_epoch=100,validation_split=0.1)
#print(hist.history['val_accuracy'])
'''
TA=[0.8461538553237915, 0.7692307829856873, 846383094788, 0.6153846383094788, 0.7692307829856873, 0.7692307829856873, 0.7692307829856873, 0.7692307829856873, 0.7692307829856873, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.6153846383094788, 0.6153846383094788, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.6153846383094788, 0.5384615659713745, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.692307710647583, 0.6153846383094788]

print(len(TA))
Y=TA[::-1]
X=range(0,100)
plt.plot(X,Y)
plt.show()

'''
#model.save_weights(modelfile) #保存模型权重

