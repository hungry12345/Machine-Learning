from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train.shape
plt.imshow(X_train[1])

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)


def classification_model():
    model = Sequential()
    model.add(Dense(num_pixels, activation = 'relu', input_shape = (num_pixels,)))
    model.add(Dense(100, activation ='relu'))
    model.add(Dense(num_classes, activation ='softmax'))

    # Compile the model
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


model = classification_model()
model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), verbose = 2)

scores = model.evaluate(X_test, y_test, verbose=0)