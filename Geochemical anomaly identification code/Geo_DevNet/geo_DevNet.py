# -*- coding: utf-8 -*-
"""
This code utilizes the DevNet network to implement anomaly identification on the training data.
Code modified from Pang, G., Shen, C., & Van Den Hengel, A. (2019, July).
Deep anomaly detection with deviation networks.
In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 353-362).

"""

import numpy as np
from sklearn.preprocessing import normalize

np.random.seed(8888)
import tensorflow as tf

tf.set_random_seed(8888)
sess = tf.Session()

from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

import argparse
import numpy as np
from geo_utils import dataLoading, aucPerformance, writeResults
from sklearn.model_selection import train_test_split

import time

MAX_INT = np.iinfo(np.int32).max


#end-to-end anomaly scoring network
def dev_network_d(input_shape):

    x_input = Input(shape=input_shape)
    intermediate = Dense(200, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01), name='hl1')(x_input)
    intermediate = Dense(80, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01), name='hl3')(intermediate)
    intermediate = Dense(20, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01), name='hl4')(intermediate)
    intermediate = Dense(1, activation='linear', name='score')(intermediate)
    return Model(x_input, intermediate)

#Gaussian Prior-based Reference Scores
#Z-Score-based Deviation Loss
def deviation_loss(y_true, y_pred):
    confidence_margin = 5.
    ref = K.variable(np.random.normal(loc=0., scale=1.0, size=5000), dtype='float32')
    dev = (y_pred - K.mean(ref)) / K.std(ref)
    inlier_loss = K.abs(dev)
    outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))
    return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

#Construct the deviation network-based detection model
def deviation_network(input_shape):
    model=dev_network_d(input_shape)
    rms = RMSprop(clipnorm=1.)
    model.compile(loss=deviation_loss, optimizer=rms)
    return model

#Batch generator
def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):

    rng = np.random.RandomState(rng.randint(MAX_INT, size=1))
    counter = 0
    while 1:
        ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)

        counter += 1
        yield (ref, training_labels)
        if (counter > nb_batch):
            counter = 0


#Batchs of samples
def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):

    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim))
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):
        if (i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = x_train[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.array(training_labels)

#Load model weight
def load_model_weight_predict(model_name, input_shape, x_test):

    model = deviation_network(input_shape)
    model.load_weights(model_name)
    scoring_network = Model(inputs=model.input, outputs=model.output)
    scores = scoring_network.predict(x_test)
    return scores

#Add anomalies to training data to replicate anomaly contaminated data sets.
def inject_noise(seed, n_out, random_seed):

    rng = np.random.RandomState(random_seed)
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace=False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace=False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise


def run_devnet(args):
    names = args.data_set.split(',')
    names=['PC8_dataset']
    random_seed = args.ramdn_seed
    for nm in names:
        runs = args.runs
        rauc = np.zeros(runs)
        ap = np.zeros(runs)
        filename = nm.strip()
        #global data_format

        x, labels = dataLoading(args.input_path + filename + ".csv")
        x=normalize(x)
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]
        n_outliers_org = outliers.shape[0]

        train_time = 0
        test_time = 0
        for i in np.arange(runs):
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2,random_state=8888,
                                                                stratify=labels)#
                                                                #,
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round ' + str(i))
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            n_outliers = len(outlier_indices)
            print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))

            n_noise = len(np.where(y_train == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            n_noise = int(n_noise)

            rng = np.random.RandomState(random_seed)
            #if data_format == 0:
            if n_outliers > args.known_outliers:
                 mn = n_outliers - args.known_outliers
                 remove_idx = rng.choice(outlier_indices, mn, replace=False)
                 x_train = np.delete(x_train, remove_idx, axis=0)
                 y_train = np.delete(y_train, remove_idx, axis=0)

            noises = inject_noise(outliers, n_noise, random_seed)
            x_train = np.append(x_train, noises, axis=0)
            y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            print('training samples num:', y_train.shape[0],
                  'outlier num:', outlier_indices.shape[0],
                  'inlier num:', inlier_indices.shape[0],
                  'noise num:', n_noise)
            n_samples_trn = x_train.shape[0]
            n_outliers = len(outlier_indices)
            print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))

            start_time = time.time()
            input_shape = x_train.shape[1:]
            epochs = args.epochs
            batch_size = args.batch_size
            nb_batch = args.nb_batch

            model = deviation_network(input_shape)
            print(model.summary())
            model_name = "./geomodel/devnet_" + filename + "_" + str(args.cont_rate) + "cr_" + str(
                args.batch_size) + "bs_" + str(args.known_outliers) + "ko_" + "d.h5"
            checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                           save_best_only=True, save_weights_only=True)

            model.fit_generator(
                batch_generator_sup(x_train, outlier_indices, inlier_indices, batch_size, nb_batch, rng),
                steps_per_epoch=nb_batch,
                epochs=epochs,
                callbacks=[checkpointer])
            train_time += time.time() - start_time

            start_time = time.time()
            scores = load_model_weight_predict(model_name, input_shape, x_test)
            test_time += time.time() - start_time
            rauc[i], ap[i] = aucPerformance(scores, y_test)

        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        train_time = train_time / runs
        test_time = test_time / runs
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))
        print("average runtime: %.4f seconds" % (train_time + test_time))
        writeResults(filename + '_' , x.shape[0], x.shape[1], n_samples_trn, n_outliers_org,
                     n_outliers,
                    mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time, path=args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="batch size used in SGD")
    parser.add_argument("--nb_batch", type=int, default=10, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=200, help="the number of epochs")
    parser.add_argument("--runs", type=int, default=10,
                        help="how many times we repeat the experiments to obtain the average performance")
    parser.add_argument("--known_outliers", type=int, default=7,
                        help="the number of labeled outliers available at hand")
    parser.add_argument("--cont_rate", type=float, default=0.015,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--input_path", type=str, default='./geodata/', help="the path of the data sets")
    parser.add_argument("--data_set", type=str, default='1', help="a list of data set names")
    parser.add_argument("--output", type=str,
                        default='./georesults/devnet_auc_performance_outliers_0.015contrate_10runs.csv',
                        help="the output file path")
    parser.add_argument("--ramdn_seed", type=int, default=8888, help="the random seed number")
    args = parser.parse_args()
    run_devnet(args)
