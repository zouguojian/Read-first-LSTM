# -- coding: utf-8 --
import tensorflow as tf
import pandas as pd
import numpy as np
import model.encoder as encoder
import model.encoder_lstm as encoder_lstm
import model.decoder as decoder
import matplotlib.pyplot as plt
import model.decoder_lstm as decoder_lstm
import model.encoder_gru as encodet_gru
import model.encoder_rnn as encoder_rnn
import os
import datetime
from model.embedding import embedding
from model.hyparameter import parameter
from model.utils import *
tf.reset_default_graph()

import model.data_next as data_load
import argparse
import csv

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"


class Model(object):
    def __init__(self, hp):
        '''
        :param para:
        '''
        self.hp = hp             # hyperparameter
        self.init_placeholder()  # init placeholder
        self.model()             # init prediction model

    def init_placeholder(self):
        '''
        :return:
        '''
        self.placeholders = {
            'features': tf.placeholder(tf.float32, shape=[None, self.hp.input_length, self.hp.features], name='input_features'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.hp.output_length], name='labels'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout')
        }

    def model(self):
        '''
        :return:
        '''
        print('#................................in the encoder step......................................#')
        with tf.variable_scope(name_or_scope='encoder'):
            encoder_init = encoder.encoder(self.placeholders['features'],
                                           self.hp.batch_size,
                                           self.hp.hidden_layer,
                                           self.hp.hidden_size,
                                           self.hp.is_training)
            (c_state, h_state) = encoder_init.encoding()
            print('encoder output shape is : ', h_state.shape)

        print('#................................in the decoder step......................................#')
        with tf.variable_scope(name_or_scope='decoder'):
            decoder_init = decoder_lstm.lstm(self.hp.batch_size,
                                             self.hp.output_length,
                                             self.hp.hidden_layer,
                                             self.hp.hidden_size,
                                             self.hp.is_training)
            self.pre = decoder_init.decoding(h_state)
            print('pres shape is : ', self.pre.shape)

        self.loss = tf.reduce_mean(
                tf.sqrt(tf.reduce_mean(tf.square(self.pre + 1e-10 - self.placeholders['labels']), axis=0)))
        self.train_op = tf.train.AdamOptimizer(self.hp.learning_rate).minimize(self.loss)

    def test(self):
        '''
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def describe(self, label, predict):
        '''
        :param label:
        :param predict:
        :return:
        '''
        plt.figure()
        # Label is observed value,Blue
        plt.plot(label[0:], 'b', label=u'actual value')
        # Predict is predicted value，Red
        plt.plot(predict[0:], 'r', label=u'predicted value')
        # use the legend
        plt.legend()
        plt.show()

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def re_current(self, a, max, min):
        return [num * (max - min) + min for num in a]

    def run_epoch(self):
        '''
        :return:
        '''
        max_rmse = 100
        self.sess.run(tf.global_variables_initializer())

        iterate = data_load.DataClass(hp=self.hp)
        train_next = iterate.next_batch(batch_size=self.hp.batch_size, epoch=self.hp.epoch, is_training=True)

        for i in range(int((iterate.length * iterate.divide_ratio - (
                iterate.input_length + iterate.output_length)) // iterate.step)
                       * self.hp.epoch // self.hp.batch_size):
            # self.hp.batch_size = 128
            # self.hp.is_training=True
            x, label = self.sess.run(train_next)
            features = np.reshape(x, [-1, self.hp.input_length, self.hp.features])
            feed_dict = construct_feed_dict(features, label, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.hp.dropout})
            loss_, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss_))
            # validate processing
            if i % 100 == 0:
                rmse_error = self.evaluate()
                if max_rmse > rmse_error:
                    print("the validate average rmse loss value is : %.6f" % (rmse_error))
                    max_rmse = rmse_error
                    self.saver.save(self.sess, save_path=self.hp.save_path + 'model.ckpt')

    def evaluate(self):
        '''
        :return:
        '''
        label_list = list()
        predict_list = list()

        # with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(self.hp.save_path)
        if not self.hp.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)

        iterate_test = data_load.DataClass(hp=self.hp)
        test_next = iterate_test.next_batch(batch_size=self.hp.batch_size, epoch=1, is_training=False)
        max, min = iterate_test.max_dict['PM2.5'], iterate_test.min_dict['PM2.5']
        print(max, min)

        for i in range(int((iterate_test.length - iterate_test.length * iterate_test.divide_ratio
                            - (iterate_test.input_length + iterate_test.output_length)) // iterate_test.output_length)
                       // self.hp.batch_size):
            x, label = self.sess.run(test_next)
            features = np.reshape(x, [-1, self.hp.input_length, self.hp.features])
            feed_dict = construct_feed_dict(features, label, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})
            pre = self.sess.run((self.pre), feed_dict=feed_dict)
            label_list.append(label)
            predict_list.append(pre)

        label_list = np.reshape(label_list, [-1, self.hp.predict_length])
        predict_list = np.reshape(predict_list, [-1, self.hp.predict_length])
        if self.hp.normalize:
            label_list = np.array(self.re_current(label_list, max, min))
            predict_list = np.array(self.re_current(predict_list, max, min))
        mae, rmse, mape, cor, r2=metric(np.round(predict_list),np.round(label_list))
        # self.describe(label_list, predict_list)   #预测值可视化
        return mae


def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    print('#......................................beginning........................................#')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    if int(val) == 1:
        para.is_training = True
    else:
        para.batch_size = 1
        para.is_training = False

    pre_model = Model(para)
    pre_model.initialize_session()

    if int(val) == 1:
        pre_model.run_epoch()
    else:
        pre_model.evaluate()

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()