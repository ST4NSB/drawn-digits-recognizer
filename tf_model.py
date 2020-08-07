import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np


class TFModel:
    class ModelsCollection:
        def CNN(self, model):
            model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation = "relu", input_shape=(28, 28, 1)))
            model.add(layers.MaxPooling2D(pool_size=(2,2)))
            model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation = "relu"))
            model.add(layers.MaxPooling2D(pool_size=(2,2)))
            model.add(layers.Flatten())
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(128, activation = "relu"))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(64, activation = "relu"))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(10))
            model.summary()
            model.compile(optimizer = "adam", loss = losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['acc'])
            print("Optimizer: ", model.optimizer)
                
                
        def MLP(self, model):
            #model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(28, 28, 1)))
            model.add(layers.Flatten(input_shape=(28, 28, 1)))
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(10))
            model.summary()
            model.compile(optimizer = "adam", loss = losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['acc'])
            print("Optimizer: ", model.optimizer)

    def __init__(self, dataset = None):
        if dataset:
            self.load_dataset(dataset)
            #print(self.train_data[0])
            self.train_data = self.normalize_dataset(self.train_data)
            self.test_data = self.normalize_dataset(self.test_data)
        self.create_empty_model()
			
    def load_dataset(self, dataset):
        (self.train_data,self.train_label),(self.test_data,self.test_label) = dataset.load_data()
	
    def normalize_dataset(self, image):
        image = image.reshape((image.shape[0], 28, 28, 1))
        #image = np.expand_dims(image, axis=0)
        image = image.astype('float32')
        image = image / 255.0
        return image
	
    def normalize_img(self, image):
        image = image.reshape((1, 28, 28, 1))
        #image = np.expand_dims(image, axis=0)
        image = image.astype('float32')
        image = image / 255.0
        return image
	
    def show_image(self, index, cmap = None):
        plt.title("Number: " + str(tfm.train_label[index]))
        plt.imshow(tfm.train_data[index], cmap=cmap)
        plt.show()
	
    def create_empty_model(self):
        self.model = models.Sequential()
	
    def add_layer_to_model(self, layer):
        self.model.add(layer)
		
    def learn(self, batch_size = 128, epochs = 10):
        self.history = self.model.fit(self.train_data, self.train_label, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

    def eval(self, show_evolution = False):
        loss, acc = self.model.evaluate(self.test_data, self.test_label, verbose=1)
        print("Loss: ", loss)
        print("Accuracy: ", acc)

        if show_evolution:
            plt.plot(self.history.history['acc'], label='accuracy')
            plt.plot(self.history.history['val_acc'], label = 'val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim([0.8, 1])
            plt.legend(loc='lower right')
            plt.show()
		
    def load_weights(self, file):
        file += ".h5"
        self.model.load_weights(file)
		
    def save_weights(self, file):
        file += ".h5"
        self.model.save_weights(file)
        
    def predict_image(self, image):
        image = self.normalize_img(image)
                
        self.ModelsCollection().CNN(self.model)
        self.load_weights(file = "trained_weights\\cnn2")
        probability_model = keras.Sequential([self.model, tf.keras.layers.Softmax()])
                
        pred = probability_model.predict(image)
        max_val = np.argmax(pred[0])

        return (max_val, pred[0][max_val])
		

if __name__ == "__main__":
    tfm = TFModel(dataset = tf.keras.datasets.mnist)
    #tfm.show_image(index = 9, cmap = 'gray')
    #print(tfm.train_data[0].size)
    tfm.ModelsCollection().CNN(tfm.model)
    #tfm.ModelsCollection().MLP(tfm.model)
    tfm.learn()
    #tfm.save_weights(file = "trained_weights\\lstm")
    #tfm.load_weights(file = "trained_weights\\cnn")
    tfm.eval()
	
	
