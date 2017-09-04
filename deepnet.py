import tensorflow as tf
import numpy as np


class NNetwork(object):

    def __init__(self, layers, name = "model"):

        # TODO: hacer que sea posible cargar el modelo sin tener que entrenarlo cada vez

        """
        :param  name: STRING name of the model
        :param layers: LIST contains, for each layer, the number of nodes in the layer
        """

        self.name = name

        # Parameters to control the state of the neural network
        self.isset = False  # Check if the neural network has been set

        self.nfeatures = None

        self.labels = []
        self.nlabels = None

        self.layers = layers

        # variable where the weights and biases of each layer will be saved (list of dictionaries {weights, biases})
        self.layerstruct = []

    def onehot(self, x):

        out = np.zeros((len(x), self.nlabels))

        for row in range(len(x)):
            index = self.labels.index(x[row + x.index[0]])
            out[row, index] = 1

        return out

    def predict(self, X):

        print("Check no reset modelo: ", self.isset)

        if not self.isset:
            print("MODEL RESET")
            self.setmodel()

        x = tf.placeholder('float', [None, len(X.columns)])

        out = self.computelayers(x)

        prediction = tf.arg_max(out, 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) # comprobar si no se resetean las variables

            saver = tf.train.import_meta_graph("model/" + self.name + ".meta")
            saver.restore(sess, tf.train.latest_checkpoint("model/"))

            pred, probs = sess.run([prediction, out], feed_dict = {x: X})
            print("Prediction: ", pred)
            print("Probabilities: ", probs)

            pred = [self.labels[i] for i in pred]

        return pred

    def setmodel(self):
        """
        function to create the model of the neural network
        """
        for layer in range(1, len(self.layers)):
            laystruct = {'weights': tf.Variable(tf.random_normal([self.layers[layer - 1], self.layers[layer]])),
                         'biases': tf.Variable(tf.random_normal([self.layers[layer]]))}

            self.layerstruct.append(laystruct)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()

            sess.run(self.layerstruct)
            saver.save(sess, "model/" + self.name)

        self.isset = True

    def computelayers(self, x):
        """
        function to predict using the neural network
        :param x: TENSOR contanins the data for the prediction
        :return: TENSOR (??) contains the values of the output layer
        """

        layers = [x]

        for lay in range(1, len(self.layers)):
            mut = tf.matmul(layers[lay - 1], self.layerstruct[lay - 1]['weights'])
            layer = tf.add(mut, self.layerstruct[lay - 1]['biases'])

            if lay != (len(self.layers) - 1):
                layer = tf.nn.relu(layer)
            layers.append(layer)

        return layers[-1]

    def train(self, X, labels):
        """
        Función para entrenar la red neuronal
        :param X: MATRIX matriz de features
        :param y: VECTOR labels
        """

        # save the number of features to be used to train the network
        self.nfeatures = len(X.columns)

        # save the number of labels and the order of labels
        self.labels = list(set(labels))
        self.nlabels = len(self.labels)

        # reshape the layers list to include the first (inputs) and last (output) layers
        self.layers = [self.nfeatures] + self.layers + [self.nlabels]

        # if the model is not set yet, initialize it
        if not self.isset:
            self.setmodel()

        x = tf.placeholder('float', [None, len(X.columns)])
        y = tf.placeholder('int64')

        # onehoty = tf.one_hot(y, depth = self.nlabels)

        prediction = self.computelayers(x)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,
                                                                      labels = y))

        optimizer = tf.train.AdamOptimizer().minimize(cost)

        # TODO: poner el número de epochs como un parámetro
        nepochs = 10
        batchsize = 100

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.import_meta_graph("model/" + self.name + ".meta")
            saver.restore(sess, tf.train.latest_checkpoint("model/"))

            for epoch in range(nepochs):
                epoch_loss = 0
                for i in range(int(len(labels) / batchsize)):
                    # TODO: entrenar el modelo en batches o online, igual es más efectivo
                    batch = list(range((i*batchsize), ((i+1)*batchsize)))
                    _, c = sess.run([optimizer, cost], feed_dict = {x: X.loc[batch, :],
                                                                    y: self.onehot(labels[batch])})
                    epoch_loss += c
                print("Epoch", (epoch + 1), "completed out of", nepochs, "Loss = ", epoch_loss)

            saver.save(sess, "model/" + self.name)

            # TODO: arreglar como se calcula la accuracy
            correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy: ', accuracy.eval({x: X, y: self.onehot(labels)}))
