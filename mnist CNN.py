# Import Modules
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
from pathlib import Path
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt
from random import randint

# Variables
rows, cols = 28, 28
batch_size = 64
num_classes = 10
epochs = 5

# Input Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalise
x_train = np.expand_dims(x_train.astype('float32'), axis=3)
x_test = np.expand_dims(x_test.astype('float32'), axis=3)
x_train /= 255
x_test /= 255
print('Train Size: {}'.format(x_train.shape[0]))
print('Test Size: {}'.format(x_test.shape[0]))

# Convert categories to onehot matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


retrain = True
if retrain:
    # Define model
    logger = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True)
    model = Sequential()

    model.add(Conv2D(32,  kernel_size=(3, 3), padding='same', input_shape=(rows, cols, 1), activation="relu"))
    model.add(Conv2D(32,  kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[logger])
    results = model.evaluate(x_test, y_test, verbose=0)
    print('Validation Loss: {} Validation Accuracy: {}'.format(results[0], results[1]))

    # Save model
    model_structure = model.to_json()
    f = Path("model_structure.json")
    f.write_text(model_structure)
    model.save_weights("model_weights.h5")

# Load model
f = Path("model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure) # Recreate keras model from json
model.load_weights("model_weights.h5")

# Input image
#img = image.load_img("test_digit.png", target_size=(rows, cols))
#test_digit = np.expand_dims(image.img_to_array(img), axis=0)

test_digit = x_train[randint(0,len(x_train)-1)]
plt.imshow(test_digit.reshape(rows,cols), cmap='gray')
plt.show()
test_digit = np.expand_dims(test_digit, axis=0)

# Prediction
prediction = model.predict(test_digit)
print('Digit: {} Confidence: {:.4f}%'.format( np.argmax(prediction[0]), 100 * prediction[0][np.argmax(prediction[0])] ))

