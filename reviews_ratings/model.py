import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import math


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  samp = tf.strings.regex_replace(lowercase, "\\\\r\\\\n", "")
  samp = tf.strings.regex_replace(samp, "\\\\r\\\\n", "")
  samp = tf.strings.regex_replace(samp, "\\\\n", "")
  samp = tf.strings.regex_replace(samp, "\\n", "")
  samp = tf.strings.regex_replace(samp, "b\'", "")
  samp = tf.strings.strip(samp)
  return samp

def vectorize_text(text, label = None):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

class Model:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

    def N_BAYES(self, size):

        prior_pr = {
            0: 0,
            1: 0
        }

        for i in self.train:
            for j in i[1]:
                prior_pr[j.numpy()] += 1

        sp = sum(prior_pr.values())

        likelihood = {
            0: dict(),
            1: dict(),
        }

        for i in self.train:
            for j in range(len(i[1])):
                key = i[1][j].numpy()
                for k in i[0][j].numpy():
                    if k == 0: continue
                    if k not in likelihood[key]:
                        likelihood[key][k] = 1
                    else:
                        likelihood[key][k] += 1

        #print(likelihood)
        correct = 0
        all = 0

        print("DONE")

        for i in self.test:
            #print(i)
            bayes_prob = []
            for key in range(0, 2):
                bayes = (prior_pr.get(key) / sp)
                for j in i[0][0].numpy():
                    if j == 0: continue
                    #print("{} -> {}".format(j, vectorize_layer.get_vocabulary()[j]))
                    if j not in likelihood[key]:
                        bayes *= (1 / size)
                    else:
                        bayes *= ((1 + likelihood[key][j]) / (size + prior_pr.get(key)))
                bayes_prob.append(bayes)


            pred = np.argmax(bayes_prob)
            real = i[1][0].numpy()

            #print(bayes_prob)
            #print(pred)
            #print(real)

            if real == pred:
                correct += 1
            all += 1

        return float(correct/all)

    def LSTM(self, units = 64, epochs = 3, optimizer = "adam"):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(max_features + 1, 64))
        model.add(tf.keras.layers.LSTM(units))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(3))

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(self.train, validation_data=self.val, epochs=epochs)
        return model.evaluate(self.test)

    def GRU(self, units = 64, epochs = 3, optimizer = "adam"):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(max_features + 1, 64))
        model.add(tf.keras.layers.GRU(units))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(3))

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(self.train, validation_data=self.val, epochs=epochs)
        return model.evaluate(self.test)


    def BIDI_LSTM(self, units = 32, epochs = 3, optimizer = "adam"):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(max_features + 1, 32))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units)))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(3))

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(self.train, validation_data=self.val, epochs=epochs)
        return model.evaluate(self.test)

    def BIDI_GRU(self, units = 64, epochs = 3, optimizer = "adam"):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(max_features + 1, 64))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(3))

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(self.train, validation_data=self.val, epochs=epochs)
        return model.evaluate(self.test)

if __name__ == "__main__":
    parent_dir = "cinemagia_reviews"

    buffer_size = 50000
    batch_size = 16
    seed = 10
    max_features = 10000
    sequence_length = 200

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        parent_dir + "/train",
        batch_size=batch_size,
        validation_split=0.1,
        subset='training',
        seed=seed)

    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        parent_dir + "/train",
        batch_size=batch_size,
        validation_split=0.1,
        subset='validation',
        seed=seed)

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        parent_dir + "/test",
        batch_size=batch_size)

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    train_ds = train_ds.shuffle(
        buffer_size, reshuffle_each_iteration=False)
    val_ds = val_ds.shuffle(
        buffer_size, reshuffle_each_iteration=False)
    test_ds = test_ds.shuffle(
        buffer_size, reshuffle_each_iteration=False)

    model = Model(train_ds, val_ds, test_ds)
    loss, acc = model.BIDI_LSTM(epochs=5)

    #loss = float(0)
    #acc = model.N_BAYES(size=len(vectorize_layer.get_vocabulary()))

    print('\nEval loss: {:.4f}, Eval accuracy: {:.4f}'.format(loss, acc))