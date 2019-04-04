import matplotlib
matplotlib.use('Agg')
import os, inspect, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def loss_record(data):

    np.save("loss", np.asarray(data))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(data)
    plt.ylabel("Cross-Entropy loss")
    plt.xlabel("Iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("loss.png")
    plt.close()

def acc_record(data):

    np.save("accuracy", np.asarray(data))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(data)
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("accuracy.png")
    plt.close()

def training(sess, neuralnet, saver, dataset, epochs, batch_size, dropout, print_step=10):

    print("\n* Training to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    train_writer = tf.summary.FileWriter(PACK_PATH+'/Checkpoint')

    epoch = 0
    start_time = time.time()
    list_loss, list_acc = [], []
    for it in range(int((dataset.amount_tr / batch_size) * epochs)):
        X_tr, Y_tr = dataset.next_batch(batch_size=batch_size, train=True)
        _ = sess.run([neuralnet.optimizer], feed_dict={neuralnet.inputs:X_tr, neuralnet.labels:Y_tr, neuralnet.dropout_prob:dropout})
        tmp_loss, tmp_acc = sess.run([neuralnet.loss, neuralnet.accuracy], feed_dict={neuralnet.inputs:X_tr, neuralnet.labels:Y_tr, neuralnet.dropout_prob:1})

        summaries = sess.run(neuralnet.summaries, feed_dict={neuralnet.inputs:X_tr, neuralnet.labels:Y_tr, neuralnet.dropout_prob:1})
        train_writer.add_summary(summaries, it)

        list_loss.append(tmp_loss)
        list_acc.append(tmp_acc)

        if(it > (dataset.amount_tr/batch_size)*epoch):
            if(epoch % print_step == 0): print("Epoch [%d / %d] \nLoss: %.5f \tAccuracy: %.5f" %(epoch, epochs, tmp_loss, tmp_acc))
            saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")
            epoch += 1

    print("Final Epoch \nLoss: %.5f \tAccuracy: %.5f" %(tmp_loss, tmp_acc))
    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    loss_record(data=list_loss)
    acc_record(data=list_acc)

def validation(sess, neuralnet, saver, dataset):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    count = 0
    confmat = np.zeros((dataset.num_class, dataset.num_class))
    while(True):
        X_te, Y_te, path_te = dataset.next_batch(batch_size=1)
        if(X_te is None): break
        preds, scores = sess.run([neuralnet.pred, neuralnet.score], feed_dict={neuralnet.inputs:X_te, neuralnet.labels:Y_te, neuralnet.dropout_prob:1})
        idx_y, idx_p = np.argmax(Y_te, axis=1)[0], preds[0]
        count += 1

        for cidx, clsname in enumerate(dataset.class_names):
            if(clsname in path_te): confmat[cidx][idx_p] += 1

    print(count)
    print("Confusion Matrix")
    print(confmat)
    print(np.sum(confmat))
