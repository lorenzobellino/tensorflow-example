import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def decode(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

def encode(text):
    tmp = [1]
    for word in text:
        if word.lower() in word_index:
            tmp.append(word_index[word.lower()])
        else:
            tmp.append(2)
    return tmp
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000);

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",maxlen=256);
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",maxlen=256);

#import the model
model = keras.models.load_model("imdb.h5")

with open("review.txt") as f:
    for line in f.readlines():
        nline = line.replace(',',' ').replace('(',' ').replace(')',' ').replace(':',' ').replace('\"',' ').replace('.',' ').replace('\n',' ').split(" ")
        enc = encode(nline)
        enc = keras.preprocessing.sequence.pad_sequences([enc], value=word_index["<PAD>"], padding="post",maxlen=256);
        pred = model.predict(enc)
        print(line)
        print(pred[0])
