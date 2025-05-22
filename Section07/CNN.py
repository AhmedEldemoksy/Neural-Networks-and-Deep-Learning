import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2️⃣ Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# 3️⃣ Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),  # Downsize to 14x14

    layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),  # Downsize to 7x7

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4️⃣ Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5️⃣ Train model
history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# 6️⃣ Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.show()

# 7️⃣ Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.2f}')
