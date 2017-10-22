"""
Build an autoencoder-like neural network with Tensorflow

Version:

"""


import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops


def Simulation(G):


    # read the init_state.txt 
    S = set()
    I = set()
    R = set()

    # choose a random infected node
    # print(G.nodes())
    S = set(G.nodes())
    rand_node = random.choice(G.nodes())
    S.remove(rand_node)
    I = set([rand_node])
    R = set()


    # implement diffusion process and write state of graph at this time point to an array in order
    numVer = len(S) + len(I) + len(R) # number of nodes
    sim = [] # expected return list of simulation process
    curState = np.array([0 for x in range(numVer)])
    curState[rand_node] = 1
    sim.append(curState)

    # util there is no more infected point
    while len(I) != 0:
        curState = np.array([0 for x in range(numVer)]) #store current state of nodes
        Ic = I.copy()
        for i in Ic:
            if len(G.neighbors(i)) != 0:
                for p in G.neighbors(i):
                    if p in S:
                        a = contamination_test(0.5)  # infection rate prob
                        if a == 1:
                            S.discard(p); I.add(p)
                            # if p in S: S.delete(p); I.add(p)
            I.discard(i); R.add(i)
        #for s in S:
        #curState[s] = 0 
        for i in I:
            curState[i] = 1
        for r in R:
            curState[r] = -1
        sim.append(curState)  # store current state of nodes
    return sim


def random_mini_batches(X, Y, mini_batch_size = 5, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[mini_batch_size * k : mini_batch_size *(k+1), :] 
        mini_batch_Y = shuffled_Y[mini_batch_size * k : mini_batch_size *(k+1), :] 
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[m - mini_batch_size * num_complete_minibatches : m, :]
        mini_batch_Y = shuffled_Y[m - mini_batch_size * num_complete_minibatches : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# For a given probability returns 1 if the test success (node is contaminated), and 0 otherwise
def contamination_test(proba) :
    a = random.random()
    if a > proba: 
        return 0
    else:
        return 1


def model(graph, num_sim, num_test, sim, starter_learning_rate = 0.01,
          num_epochs = 2000, print_cost = True):
    # Parameters
    # m = np.array(reSim).shape[0]
    num_input = np.array(sim).shape[1]
    costs =[]



    # Network Parameters
    num_hidden_1 = 256 # 1st layer num features
    num_hidden_2 = 64 # 2nd layer num features
    num_hidden_3 = 32  # 3nd layer num features

    # tf Graph input 
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_input])


    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2])),
        'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
        'decoder_h3': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
        'decoder_b1': tf.Variable(tf.random_normal([num_hidden_2])),
        'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
        'decoder_b3': tf.Variable(tf.random_normal([num_input])),
    }


    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                       biases['encoder_b3']))


        return layer_3


    # Building the decoder
    def decoder(x):
        # Decoder Hidden layer with sigmoid activation #1


        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                       biases['decoder_b3']))

        a = tf.fill([1, num_input], -0.5)
        b = tf.fill([1, num_input], 0.5)


        # https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators
        layer_3_mdf = tf.div(tf.add(layer_3, a), b)
        return layer_3_mdf


    


    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)
    # a = tf.constant(0.5, decoder_op.shape)
    # b = tf.constant(2.0)
    # # Prediction
    # y_pred =tf.matmul((decoder_op - a), b)
    y_pred = decoder_op

    def getyp(y_p):
        for i in range(y_p.shape[0]):
            for j in range(y_p.shape[1]):
                if y_p[i][j] > 0.5:
                    y_p[i][j] = 1
                elif y_p[i][j] < -0.5:
                    y_p[i][j] = -1
                else:
                    y_p[i][j] = 0

        return y_p


    # Targets (Labels)
    y_true = Y

    # Define loss and optimizer, minimize the squared error
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))



    # learning_rate decay scheme...
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                        10000, 0.95, staircase=False)
            
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)


    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()



    # Start Training
    # Start a new TF session
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            # num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            # minibatches = random_mini_batches(np.array(reSim)[0:m-1, :], np.array(reSim)[1:m, :], minibatch_size)
            for x in range(num_sim):
                
                sim = Simulation(graph)
                m = np.array(sim).shape[0]
                _ , sim_cost = sess.run([optimizer, loss], feed_dict={X:  np.array(sim)[0:m-1, :], Y: np.array(sim)[1:m, :]})
                epoch_cost += sim_cost / num_sim

            # for minibatch in minibatches:

            #     # Select a minibatch
            #     (minibatch_X, minibatch_Y) = minibatch
            
            #     # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
            #     _ , minibatch_cost = sess.run([optimizer, loss], feed_dict={X:  minibatch_X, Y: minibatch_Y})
            
                
            #     epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

    

        #     #start test:
        #     test_sim = Simulation(graph)
        #     print(np.shape(np.array(test_sim)))
        #     L = len(test_sim)
        #     t = np.array(test_sim)[0, :]
        #     print(np.shape(t))
        #     print(np.shape(np.reshape(t, (1, 100))))

        #     t = np.reshape(t, (1, 100))
        #     test_cost = 0
        #     #sess.run(init)
        #         g = sess.run(loss, feed_dict={X: t, Y: np.reshape(np.array(test_sim)[l, :],(1,100))}) 
        #         # test_cost += tf.pow(g - test_sim[l], 2)  
        #         test_cost += g   
        #         t = y_pred; l=l+1
        #         #'The value of a feed cannot be a tf.Tensor object. '
        #     loss = test_cost/(L-1)
        #     print(loss)

        #start test:
        
        test_acc_avg = 0
        test_cost_avg = 0
        for x in range(num_test):
            test_sim = Simulation(graph)
            n = np.array(test_sim).shape[0]
            test_cost, y_p = sess.run([loss,y_pred], feed_dict={X: np.array(test_sim)[0:n-1, :], Y: np.array(test_sim)[1:n, :]})
            yp = getyp(y_p)
            # print(yp)
            y = tf.constant(yp)

            accuracy = tf.cast(tf.equal(y_true, y), tf.int32)
            test_accuracy = sess.run(accuracy, feed_dict={X: np.array(test_sim)[0:n-1, :], Y: np.array(test_sim)[1:n, :]})
            # print(test_accuracy)
            test_cost_avg += test_cost/num_test
            test_acc_avg += np.mean(test_accuracy)/num_test
        print(test_acc_avg)
        print(test_cost_avg)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
