import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data();

class_names = ['T.shirt/top', 'Trousers', 'Pullover', 'Dress' , 'Coat',
                'Snadal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

print(train_images[1])
plt.imshow(train_images[1],cmap = plt.cm.binary)
plt.show()
