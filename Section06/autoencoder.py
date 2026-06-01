import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape

# 1️ Load data
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2️ Build Autoencoder (Compression)
input_img = Input(shape=(28, 28))

# Encoder (Compression)
x = Flatten()(input_img)              # 28x28 → 784
encoded = Dense(32, activation='relu')(x)   # 784 → 32   compression

# Decoder (Reconstruction)
decoded = Dense(784, activation='sigmoid')(encoded)
output_img = Reshape((28, 28))(decoded)

# Model
autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 3️ Train
autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

# 4️ Get compressed representation
encoder = Model(input_img, encoded)
compressed_imgs = encoder.predict(x_test)

print("Compressed shape:", compressed_imgs.shape)

# 5️ Reconstruct images
decoded_imgs = autoencoder.predict(x_test)

# 6️ Show results
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # original
    plt.subplot(2, n, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # reconstructed
    plt.subplot(2, n, i+n+1)
    plt.imshow(decoded_imgs[i], cmap='gray')
    plt.title("Compressed")
    plt.axis('off')

plt.show()
