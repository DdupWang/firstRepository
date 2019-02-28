import tensorflow as tf
import numpy as np

#神经网络层
def add_layer(inputs, in_size, out_size, activation_function=None,biases=0.1):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + biases)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#神经网络训练
def trainCNN(eci,inputs,outputs,inputSize,numClass,size):
    g = tf.Graph()
    with g.as_default():
        nnX = tf.placeholder(tf.float32, shape=[None, inputSize],name="x")
        nny = tf.placeholder(tf.float32, shape=[None, numClass],name="y")
        hiddenSize=10;
        #输入层
        input_layer=add_layer(nnX,inputSize,hiddenSize,tf.nn.relu)
        #隐藏层
        hidden_layer=add_layer(input_layer,hiddenSize,hiddenSize,tf.nn.relu)
        #输出层
        output_layer=add_layer(hidden_layer,hiddenSize,numClass,biases=-100.0)
        output_layer=tf.add(output_layer,tf.zeros([1, numClass]),name="y_pre")
        loss = tf.reduce_mean(tf.square(output_layer - nny))
        optimizer = tf.train.GradientDescentOptimizer(0.05)
        train = optimizer.minimize(loss)

        modelPath=os.path.join("C:\\Users\\Administrator\\PycharmProjects\\myPostitionModel\\modeldata",str(eci)+".ckpt")

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i in range(1000):
            rand_index = np.random.choice(size, size=int(size/10))
            inputsx=inputs[rand_index]
            outputsy=outputs[rand_index]
            sess.run(train,feed_dict={nnX:inputsx,nny:outputsy})
            if(i%100==0):
                lossvalue=sess.run(loss,feed_dict={nnX:inputsx,nny:outputsy})
                print(lossvalue)
        saver.save(sess, modelPath)
        return lossvalue