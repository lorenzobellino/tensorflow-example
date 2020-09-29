import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def decode(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# print(decode(train_data[0]))


data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000);

#print(train_data[0])

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",maxlen=256);
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",maxlen=256);

#definizione del modello

#output è solo uno perchè vogliamo sapere se la review è positiva oppure negativa
#il layer di embedding cerca di assimilare dati simili in questo caso cerca di capire quali parole hanno un simile significato
#10000 = creo diecimila word vector e cerca di raggrupparli in modo che i vettori di parole simili siano vicini
#16 sono i coefficienti del vettore
#global average pooling serve per ridurre la dimensione del dataset visto che 16 dimensioni sono troppe in  questo caso
#
#16 nei dense neuron è un numero preso a caso possiamo cambiarlo
#e l'ultimo livello denso è solo un neurone che manda in output un valore [0,1]

model = keras.Sequential([
    keras.layers.Embedding(88000, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16,activation = "relu"),
    keras.layers.Dense(1,activation = "sigmoid")
])

#train the model

model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])

#validation data, splitto training data in due per costruire la validation data
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

#batch size = quante reviews vengono caricate ogni volta
fitModel = model.fit(x_train, y_train, epochs = 40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

result = model.evaluate(test_data, test_labels)

print(result)

index = 1
test = test_data[index]
predict = model.predict([test])
print("review: ")
print(decode(test))
print("Prediction: "+str(predict[index]))
print("Actual    : "+str(test_labels[index]))
print(result)

#PER SALVARE IL MODELLO BASTA:

model.save("imdb.h5")
