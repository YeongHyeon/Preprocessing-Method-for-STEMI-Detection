import argparse

import tensorflow as tf

import source.neuralnet_resnet34 as nn
import source.datamanager as dman
import source.tf_process as tfp

def main():

    dataset = dman.DataSet(setname=FLAGS.setname, tr_ratio=FLAGS.tr_ratio)

    neuralnet = nn.ConvNet(data_dim=dataset.data_dim, channel=dataset.channel, num_class=dataset.num_class, learning_rate=FLAGS.lr)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    tfp.training(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, dropout=FLAGS.dropout)
    tfp.validation(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='Number of epoch for training')
    parser.add_argument('--batch', type=int, default=200, help='Mini-batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--dropout', type=float, default=1, help='Dropout ratio for training.')
    parser.add_argument('--setname', type=str, default="dataset_BP", help='Name of dataset for use.')
    parser.add_argument('--tr_ratio', type=float, default=0.9, help='Ratio of patient for training to total patient.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
