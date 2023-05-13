
import tensorflow as tf
#import numpy as np

'''
Készíts egy metódust ami a cifar100 adatbázisból betölti a train és test adatokat. (tf.keras.datasets.cifar100.load_data())
Majd a tanitó, és tesztelő adatokat normalizálja, és vissza is tér velük.


Egy példa a kimenetre: train_images, train_labels, test_images, test_labels
függvény neve: cifar100_data
'''

def cifar100_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
  x_train = x_train.astype('float32') / 255
  y_train = y_train.astype('float32') / 255
  return x_train, y_train, x_test, y_test


#train_images, train_labels, test_images, test_labels = cifar100_data()
#print(train_images, train_labels, test_images, test_labels)

'''
Készíts egy konvolúciós neurális hálót, ami képes felismerni a képen mi van a 100 osztály közül.
A háló kimenete legyen 100 elemű, és a softmax aktivációs függvényt használja.
Hálon belül tetszőleges számú réteg lehet..


Egy példa a kimenetre: model,
return type: keras.engine.sequential.Sequential
függvény neve: cifar100_model
'''

def cifar100_model(): #-> tf.keras.engine.sequential.Sequential:
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(100, activation='softmax')
  ])
  return model

#model = cifar100_model()


#from keras.backend import sparse_categorical_crossentropy
'''
Készíts egy metódust, ami a bemeneti hálot compile-olja.
Optimizer: Adam
Loss: SparseCategoricalCrossentropy(from_logits=False)

Egy példa a bemenetre: model
Egy példa a kimenetre: model
return type: keras.engine.sequential.Sequential
függvény neve: model_compile
'''

def model_compile(model):
  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
  return model

#compiled_model = model_compile(model)


'''
Készíts egy metódust, ami a bemeneti hálót feltanítja.

Egy példa a bemenetre: model,epochs, train_images, train_labelsz
Egy példa a kimenetre: model
return type: keras.engine.sequential.Sequential
függvény neve: model_fit
'''

def model_fit(model, epochs, train_images, train_labels):
  model.fit(train_images, train_labels, epochs=epochs, batch_size=64)
  return model

#fmodel= model_fit(compiled_model, 10, train_images, train_labels)


'''
Készíts egy metódust, ami a bemeneti hálót kiértékeli a teszt adatokon.

Egy példa a bemenetre: model, test_images, test_labels
Egy példa a kimenetre: test_loss, test_acc
return type: float, float
függvény neve: model_evaluate
'''

def model_evaluate(model, test_images, test_labels):
  return model.evaluate(test_images, test_labels)

#test_loss, test_acc = model_evaluate(fmodel, test_images, test_labels)
#print(test_loss)
#print( test_acc)