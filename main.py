import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Cargar datos de MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar los datos
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convertir etiquetas a categorías one-hot
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential([
  Flatten(input_shape=(28, 28)), # Capa para aplanar la entrada
  Dense(128, activation='relu'), # Capa densa con 128 neuronas
  Dense(10, activation='softmax') # Capa de salida con 10 neuronas para las 10 clases
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=32)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)
