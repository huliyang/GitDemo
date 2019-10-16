import numpy as np
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils

import matplotlib.pyplot as plt
import matplotlib.image as processimage

#Load mnist RAW dataset拉取原始数据
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

#Prepare准备数据
#reshape
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)
#-set type into float32设置成浮点型
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255
X_test /= 255

#Prepare basic setups
batch_size = 1024
nb_class = 10
nb_epochs = 4

#Class vectors
Y_test = np_utils.to_categorical(Y_test,nb_class)
Y_train = np_utils.to_categorical(Y_train,nb_class)

#设置网络结构
model = Sequential()
#1st layer
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))#overfit

#2nd layer
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#3rd layer
model.add(Dense(10))
model.add(Activation('softmax'))

#编译 Complied
model.compile(
    loss = "categorical_crossentropy",
    optimizer = "rmsprop",
    metrics = ['accuracy'],
)

#启动网络训练 Fire upl
Trainning = model.fit(
    X_train,Y_train,
    batch_size = batch_size,
    epochs = nb_epochs,
    validation_data = (X_test,Y_test),
    #verbose = 2
)
#Trainning.history
#Trainning.params

#拉取test里的图
testrun = X_test[9999].reshape(1,784)
testlabel = Y_test[9999]
plt.imshow(testrun.reshape([28,28]))

#判断输出结果
pred = model.predict(testrun)
print(testrun)
print("label of test same Y_test[9999]-->",testlabel)
print("预测结果：",pred)
print([final.argmax() for final in pred])

"""
#用自己的图预测
target_img = processimage.imread('path')
target_img = target_img.reshape(1,784)
target_img = np.array(target_img)

target_img = target_img.astype("float32")
target_img /= 255
mypred = model.predict(target_img)
print(myfinal.argmax() for myfinal in mypred)

"""

