import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot = True)

# define the structure of the neural network
# first argument, size of the data
# last argument, number of classes
# intermediate arguments, number of nodes of each layer
nnetwork = [784, 500, 500, 500, 10]

# length of the batch in which the data is loaded into memory
batchsize = 100

# placeholders for tensorflow
x = tf.placeholder('float', [None, nnetwork[0]])
y = tf.placeholder('float')

def nnmodel(data):
    # store the weights and biases of each layer
    layerstruct = []
    layers = [data]

    for lay in range(1, len(nnetwork)):
        laystruct = {'weights': tf.Variable(tf.random_normal([nnetwork[lay - 1], nnetwork[lay]])),
                     'biases': tf.Variable(tf.random_normal([nnetwork[lay]]))}

        layerstruct.append(laystruct)

        layer = tf.add(tf.matmul(layers[lay - 1], layerstruct[lay - 1]['weights']), layerstruct[lay - 1]['biases'])
        if lay != (len(nnetwork)-1):
            layer = tf.nn.relu(layer)
        layers.append(layer)

    return layers[-1]


def nntrain(x):

    prediction = nnmodel(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    nepochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(nepochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batchsize)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size = batchsize)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c

            print("Epoch", epoch, "completed out of", nepochs, "Loss = ", epoch_loss)

        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

nntrain(x)
