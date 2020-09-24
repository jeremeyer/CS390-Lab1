
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from sklearn import metrics as sk


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return np.exp(-x) / ((np.exp(-x) + 1)**2)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        if minibatches == False:
            #for i in range(0, epochs):
                # one passthrough of entire data
                #for j in range(0, xVals.shape[0]):
                for j in range(epochs):
                    oLayer1, oLayer2 = self.__forward(xVals[j])

                    #print("mse: %f" % np.square(yVals[j] - oLayer2).mean())

                    l2e = yVals[j] - oLayer2                                 # (dc/do) per output neuron
                    l2d = np.multiply(l2e, self.__sigmoidDerivative(oLayer2))  # (do/dw2)
                    l1e = np.dot(l2d, np.transpose(self.W2))                # (dw2/do1)
                    l1d = np.multiply(l1e, self.__sigmoidDerivative(oLayer1))  # (do1/dw1)

                    l1a = np.dot(xVals[j].reshape(xVals[j].shape[0], -1), np.transpose(l1d.reshape(l1d.shape[0], -1))) * self.lr
                    l2a = np.dot(oLayer1.reshape(oLayer1.shape[0], -1), np.transpose(l2d.reshape(l2d.shape[0], -1))) * self.lr

                    self.W1 = self.W1 + l1a
                    self.W2 = self.W2 + l2a
        else:
            batch_gen = self.__batchGenerator(xVals, mbs)
            batch_out_gen = self.__batchGenerator(yVals, mbs)
            for i in range(epochs):
                try:
                    batch_in = next(batch_gen)
                    batch_out = next(batch_out_gen)
                except StopIteration:
                    batch_gen = self.__batchGenerator(xVals, mbs)
                    batch_out_gen = self.__batchGenerator(yVals, mbs)
                    batch_in = next(batch_gen)
                    batch_out = next(batch_out_gen)
                for j in range(len(batch_in)):
                    data = batch_in[j]
                    dataOut = batch_out[j]
                    oLayer1, oLayer2 = self.__forward(data)

                    l2e = dataOut - oLayer2                                 # (dc/do) per output neuron
                    l2d = np.multiply(l2e, self.__sigmoidDerivative(oLayer2))  # (do/dw2)
                    l1e = np.dot(l2d, np.transpose(self.W2))                # (dw2/do1)
                    l1d = np.multiply(l1e, self.__sigmoidDerivative(oLayer1))  # (do1/dw1)

                    l1a = np.dot(data.reshape(data.shape[0], -1), np.transpose(l1d.reshape(l1d.shape[0], -1))) * self.lr
                    l2a = np.dot(oLayer1.reshape(oLayer1.shape[0], -1), np.transpose(l2d.reshape(l2d.shape[0], -1))) * self.lr

                    self.W1 = self.W1 + l1a
                    self.W2 = self.W2 + l2a
                pass
            pass


    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #range reduction here (0-255 ==> 0.0-1.0).

    xTrain = xTrain / 255.
    xTest = xTest / 255.

    xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2])
    xTest = xTest.reshape(xTest.shape[0], xTest.shape[1] * xTest.shape[2])

    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        # 512 neuron neural network
        model = NeuralNetwork_2Layer(inputSize=784, outputSize=10, neuronsPerLayer=512)
        model.train(xTrain, yTrain, epochs = 1000, mbs = 100)
        #model.train()
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        inputs = tf.keras.Input(shape=(784,))
        layer = tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid)(inputs)
        dropLayer1 = tf.keras.layers.Dropout(0.2)(layer)
        layer2 = tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid)(dropLayer1)
        dropLayer2 = tf.keras.layers.Dropout(0.2)(layer2)
        outputs = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)(dropLayer2)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=tf.keras.losses.MeanSquaredError())
        model.fit(x=xTrain, y=yTrain, epochs=20, batch_size = 64)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        out = np.empty(shape=(data.shape[0], 10))
        for i in range(0, len(data)):
            out[i] = model.predict(data[i])
        return out
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        out = model.predict(data)
        return out
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    pred_list = np.empty(shape=(preds.shape[0],))
    true_list = np.empty(shape=(yTest.shape[0],))
    for i in range(preds.shape[0]):
        pred_list[i] = np.argmax(preds[i])
        true_list[i] = np.argmax(yTest[i])
        #print(preds[i])
        #if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        if (np.argmax(preds[i]) == np.argmax(yTest[i])): acc = acc + 1
    accuracy = acc / preds.shape[0]
    conf_mat = sk.confusion_matrix(true_list, pred_list)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("confusion matrix: \n%s\n" % conf_mat)
    true_list_str = []
    pred_list_str = []
    for i in range(true_list.shape[0]):
        true_list_str.append(str(true_list[i]))
        pred_list_str.append(str(pred_list[i]))
    print("f1 score: %s" % sk.f1_score(true_list_str, pred_list_str, average='macro'))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
