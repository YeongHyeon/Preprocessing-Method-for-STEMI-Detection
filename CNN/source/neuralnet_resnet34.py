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

        conv1_1 = self.convolution(inputs=self.inputs, filters=64, k_size=7, stride=1, padding="SAME")
        pool1 = self.maxpool(inputs=conv1_1, pool_size=2)

        conv2_1 = self.convolution(inputs=pool1, filters=64, k_size=3, stride=1, padding="SAME")
        conv2_2 = self.convolution(inputs=conv2_1, filters=64, k_size=3, stride=1, padding="SAME")
        conv2_concat = pool1 + conv2_2

        conv3_1 = self.convolution(inputs=conv2_concat, filters=64, k_size=3, stride=1, padding="SAME")
        conv3_2 = self.convolution(inputs=conv3_1, filters=64, k_size=3, stride=1, padding="SAME")
        conv3_concat = conv2_concat + conv3_2

        conv4_1 = self.convolution(inputs=conv3_concat, filters=64, k_size=3, stride=1, padding="SAME")
        conv4_2 = self.convolution(inputs=conv4_1, filters=64, k_size=3, stride=1, padding="SAME")
        conv4_concat = conv3_concat + conv4_2

        """=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""
        pool2 = self.maxpool(inputs=conv4_concat, pool_size=2)

        conv5_1 = self.convolution(inputs=pool2, filters=128, k_size=3, stride=1, padding="SAME")
        conv5_2 = self.convolution(inputs=conv5_1, filters=128, k_size=3, stride=1, padding="SAME")
        conv5_res = self.convolution(inputs=pool2, filters=128, k_size=1, stride=1, padding="SAME")
        conv5_concat = conv5_res + conv5_2

        conv6_1 = self.convolution(inputs=conv5_concat, filters=128, k_size=3, stride=1, padding="SAME")
        conv6_2 = self.convolution(inputs=conv6_1, filters=128, k_size=3, stride=1, padding="SAME")
        conv6_concat = conv5_concat + conv6_2

        conv7_1 = self.convolution(inputs=conv6_concat, filters=128, k_size=3, stride=1, padding="SAME")
        conv7_2 = self.convolution(inputs=conv7_1, filters=128, k_size=3, stride=1, padding="SAME")
        conv7_concat = conv6_concat + conv7_2

        conv8_1 = self.convolution(inputs=conv7_concat, filters=128, k_size=3, stride=1, padding="SAME")
        conv8_2 = self.convolution(inputs=conv8_1, filters=128, k_size=3, stride=1, padding="SAME")
        conv8_concat = conv7_concat + conv8_2

        """=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""
        pool3 = self.maxpool(inputs=conv8_concat, pool_size=2)

        conv9_1 = self.convolution(inputs=pool3, filters=256, k_size=3, stride=1, padding="SAME")
        conv9_2 = self.convolution(inputs=conv9_1, filters=256, k_size=3, stride=1, padding="SAME")
        conv9_res = self.convolution(inputs=pool3, filters=256, k_size=1, stride=1, padding="SAME")
        conv9_concat = conv9_res + conv9_2

        conv10_1 = self.convolution(inputs=conv9_concat, filters=256, k_size=3, stride=1, padding="SAME")
        conv10_2 = self.convolution(inputs=conv10_1, filters=256, k_size=3, stride=1, padding="SAME")
        conv10_concat = conv9_concat + conv10_2

        conv11_1 = self.convolution(inputs=conv10_concat, filters=256, k_size=3, stride=1, padding="SAME")
        conv11_2 = self.convolution(inputs=conv11_1, filters=256, k_size=3, stride=1, padding="SAME")
        conv11_concat = conv10_concat + conv11_2

        conv12_1 = self.convolution(inputs=conv11_concat, filters=256, k_size=3, stride=1, padding="SAME")
        conv12_2 = self.convolution(inputs=conv12_1, filters=256, k_size=3, stride=1, padding="SAME")
        conv12_concat = conv11_concat + conv12_2

        conv13_1 = self.convolution(inputs=conv12_concat, filters=256, k_size=3, stride=1, padding="SAME")
        conv13_2 = self.convolution(inputs=conv13_1, filters=256, k_size=3, stride=1, padding="SAME")
        conv13_concat = conv12_concat + conv13_2

        conv14_1 = self.convolution(inputs=conv13_concat, filters=256, k_size=3, stride=1, padding="SAME")
        conv14_2 = self.convolution(inputs=conv14_1, filters=256, k_size=3, stride=1, padding="SAME")
        conv14_concat = conv13_concat + conv14_2

        """=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""
        pool4 = self.maxpool(inputs=conv14_concat, pool_size=2)

        conv15_1 = self.convolution(inputs=pool4, filters=512, k_size=3, stride=1, padding="SAME")
        conv15_2 = self.convolution(inputs=conv15_1, filters=512, k_size=3, stride=1, padding="SAME")
        conv15_res = self.convolution(inputs=pool4, filters=512, k_size=1, stride=1, padding="SAME")
        conv15_concat = conv15_res + conv15_2

        conv16_1 = self.convolution(inputs=conv15_concat, filters=512, k_size=3, stride=1, padding="SAME")
        conv16_2 = self.convolution(inputs=conv16_1, filters=512, k_size=3, stride=1, padding="SAME")
        conv16_concat = conv15_concat + conv16_2

        conv17_1 = self.convolution(inputs=conv16_concat, filters=512, k_size=3, stride=1, padding="SAME")
        conv17_2 = self.convolution(inputs=conv17_1, filters=512, k_size=3, stride=1, padding="SAME")
        conv17_concat = conv16_concat + conv17_2

        pool5 = self.avgpool(inputs=conv17_concat, pool_size=2)

        flat = self.flatten(inputs=pool5)

        fc1 = self.fully_connected(inputs=flat, num_outputs=self.num_class, activate_fn=None)

        return fc1

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

    def avgpool(self, inputs=None, pool_size=2):
        avgp = tf.layers.average_pooling1d(inputs=inputs, pool_size=pool_size, strides=pool_size, padding='SAME', data_format='channels_last', name=None)

        print("Avg Pool: "+str(avgp.shape))
        return avgp

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
