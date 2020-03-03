import numpy as np
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras_self_attention import SeqSelfAttention


class Generation:
    def __init__(self, rc, n_unit, epoch):
        self.n_unit = n_unit
        self.epoch = epoch
        self.words = ". ".join(rc).split(" ")
        str_to_n = {str: n for n, str in enumerate(self.words)}
        self.n_to_str = {n: str for n, str in enumerate(self.words)}
        self.X = []
        Y = []
        for i, nexti in zip(self.words, self.words[1::]):
            sequence = i
            label = nexti
            self.X.append(str_to_n[sequence])
            Y.append(str_to_n[label])
        self.X_modified = np.reshape(self.X, (len(self.X), 1, 1)) / float(len(self.words))
        self.Y_modified = np_utils.to_categorical(Y)

    def getRF_modelA(self):
        model = Sequential()
        model.add(Bidirectional(GRU(self.n_unit, input_shape=(self.X_modified.shape[1], self.X_modified.shape[2]), return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(GRU(self.n_unit))
        model.add(Dropout(0.2))
        model.add(Dense(self.Y_modified.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(self.X_modified, self.Y_modified, epochs=self.epoch, batch_size=50)
        return self.getRF(model)

    def getRF_modelB(self):
        model = Sequential()
        model.add(GRU(self.n_unit, input_shape=(self.X_modified.shape[1], self.X_modified.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(self.n_unit))
        model.add(Dropout(0.2))
        model.add(Dense(self.Y_modified.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(self.X_modified, self.Y_modified, epochs=self.epoch, batch_size=50)
        return self.getRF(model)

    def getRF_modelC(self):
        model = Sequential()
        model.add(Bidirectional(GRU(self.n_unit, input_shape=(self.X_modified.shape[1], self.X_modified.shape[2]), return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(self.n_unit)))
        model.add(Dropout(0.2))
        model.add(Dense(self.Y_modified.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(self.X_modified, self.Y_modified, epochs=self.epoch, batch_size=50)
        return self.getRF(model)

    def getRF(self, model):
        string_mapped = self.X[:1]
        full_string = [self.n_to_str[value] for value in string_mapped]
        for i in range(10):
            x = np.reshape(string_mapped, (1, len(string_mapped), 1)) / float(len(self.words))
            pred_index = np.argmax(model.predict(x, verbose=0))
            full_string.append(self.n_to_str[pred_index])
            string_mapped.append(pred_index)
            string_mapped = string_mapped[1:len(string_mapped)]
        return " ".join(full_string)