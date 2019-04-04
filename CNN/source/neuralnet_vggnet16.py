import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

class ConvNet(object):

    def __init__(self, data_dim, channel, num_class, learning_rate):

        print("\n** Initialize CNN Layers")
        self.num_class = num_class
        self.inputs = tf.placeholder(tf.float32, [None, data_dim, channel])
        self.labels = tf.placeholder(tf.float32, [None, self.num_class])
        self.dropout_prob = tf.placeholder(tf.float32, shape=[])
        print("Input: "+str(self.inputs.shape))

        fc = self.convnet_module()
        self.score = tf.nn.softmax(fc)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.pred = tf.argmax(self.score, 1)
        self.correct = tf.equal(self.pred, tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summaries = tf.summary.merge_all()

    def convnet_module(self):

        conv1_1 = self.convolution(inputs=self.inputs, filters=64, k_size=3, stride=1, padding="SAME")
        conv1_2 = self.convolution(inputs=conv1_1, filters=64, k_size=3, stride=1, padding="SAME")
        pool1 = self.maxpool(inputs=conv1_2, pool_size=2)

        conv2_1 = self.convolution(inputs=pool1, filters=128, k_size=3, stride=1, padding="SAME")
        conv2_2 = self.convolution(inputs=conv2_1, filters=128, k_size=3, stride=1, padding="SAME")
        pool2 = self.maxpool(inputs=conv2_2, pool_size=2)

        conv3_1 = self.convolution(inputs=pool2, filters=256, k_size=3, stride=1, padding="SAME")
        conv3_2 = self.convolution(inputs=conv3_1, filters=256, k_size=3, stride=1, padding="SAME")
        conv3_3 = self.convolution(inputs=conv3_2, filters=256, k_size=3, stride=1, padding="SAME")
        pool3 = self.maxpool(inputs=conv3_3, pool_size=2)

        conv4_1 = self.convolution(inputs=pool3, filters=512, k_size=3, stride=1, padding="SAME")
        conv4_2 = self.convolution(inputs=conv4_1, filters=512, k_size=3, stride=1, padding="SAME")
        conv4_3 = self.convolution(inputs=conv4_2, filters=512, k_size=3, stride=1, padding="SAME")
        pool4 = self.maxpool(inputs=conv4_3, pool_size=2)

        conv5_1 = self.convolution(inputs=pool4, filters=512, k_size=3, stride=1, padding="SAME")
        conv5_2 = self.convolution(inputs=conv5_1, filters=512, k_size=3, stride=1, padding="SAME")
        conv5_3 = self.convolution(inputs=conv5_2, filters=512, k_size=3, stride=1, padding="SAME")
        pool5 = self.maxpool(inputs=conv5_3, pool_size=2)

        flat = self.flatten(inputs=pool5)

        fc1 = self.fully_connected(inputs=flat, num_outputs=4096, activate_fn=tf.nn.relu)
        drop1 = tf.nn.dropout(fc1, keep_prob=self.dropout_prob)
        fc2 = self.fully_connected(inputs=drop1, num_outputs=4096, activate_fn=tf.nn.relu)
        drop2 = tf.nn.dropout(fc2, keep_prob=self.dropout_prob)
        fc3 = self.fully_connected(inputs=drop2, num_outputs=self.num_class, activate_fn=None)

        return fc3

    def convolution(self, inputs=None, filters=32, k_size=3, stride=1, padding="SAME"):

        xavier = tf.contrib.layers.xavier_initializer()

        conv = tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=k_size, strides=1,
        padding=padding, data_format='channels_last', dilation_rate=1,
        activation=tf.nn.relu, use_bias=True,
        kernel_initializer=tf.contrib.keras.initializers.he_normal(), bias_initializer=tf.contrib.keras.initializers.he_normal(),
        kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, trainable=True, name=None, reuse=None)

        print("Convolution: "+str(conv.shape))
        return conv

    def relu(self, inputs=None):

        re = tf.nn.relu(features=inputs, name=None)

        print("ReLU: "+str(re.shape))
        return re

    def maxpool(self, inputs=None, pool_size=2):

        maxp = tf.layers.max_pooling1d(inputs=inputs, pool_size=pool_size, strides=pool_size, padding='SAME', data_format='channels_last', name=None)

        print("Max Pool: "+str(maxp.shape))
        return maxp

    def flatten(self, inputs=None):

        flat = tf.contrib.layers.flatten(inputs=inputs)

        print("Flatten: "+str(flat.shape))
        return flat

    def fully_connected(self, inputs=None, num_outputs=None, activate_fn=None):

        full_con = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=num_outputs,
        activation_fn=activate_fn, normalizer_fn=None, normalizer_params=None,
        weights_initializer=tf.contrib.keras.initializers.he_normal(), weights_regularizer=None,
        biases_initializer=tf.contrib.keras.initializers.he_normal(), biases_regularizer=None, reuse=None,
        variables_collections=None, outputs_collections=None, trainable=True, scope=None)

        print("Fully Connected: "+str(full_con.shape))
        return full_con
