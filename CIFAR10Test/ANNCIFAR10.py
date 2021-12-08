import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
#Load the dataset:Split the dataset to train and test
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("x_train.shape:",x_train.shape)
print("x_test.shape:",x_test.shape)
y_train.reshape(-1,)
y_train[:1]
#Separate features and Labels
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
sample = np.random.choice(np.arange(50000),10)
fig, axes = plt.subplots(2, 5, figsize=(12,4))
axes = axes.ravel()
for i in range(10):
    idx = sample[i]
    axes[i].imshow(x_train[idx])
    axes[i].set_title(labels[y_train[idx][0]])
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)
#Visualize some images
for i in range(6):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i])
plt.show()
print('Nummber of images - ',x_train.shape[0])
print('Dimensions of an image - ',x_train.shape[1:3])
print('Number of channels - ',x_train.shape[-1])
#Data Preprocessing
# Normalizing the data
x_train = x_train/255.0
x_test = x_test/255.0
#One hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
#Build a ANN model
len_flatten = np.product(x_train.shape[1:])
x_train_flatten = x_train.reshape(x_train.shape[0],len_flatten)
x_test_flatten = x_test.reshape(x_test.shape[0],len_flatten)
#Using optimiser ADAM,Activation funcation - Relu & Softmax
model = Sequential()
model.add(Dense(units=512, activation='relu', kernel_initializer='uniform',input_shape=(len_flatten,)))
model.add(Dense(units=128, activation='relu',kernel_initializer='uniform'))
model.add(Dense(units=64, activation='relu',kernel_initializer='uniform'))
model.add(Dense(units=32, activation='relu',kernel_initializer='uniform'))
model.add(Dense(units=num_classes, activation='softmax',kernel_initializer='uniform'))
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_flatten, y_train, epochs=10,validation_split=.3)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training','validation'], loc='best')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='best')
plt.show()
model.evaluate(x_test_flatten,y_test)
model.evaluate(x_train_flatten,y_train)
