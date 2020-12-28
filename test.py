import tensorflow as tf
import pandas as pd
import trainSet as trainSet
import testSet as testSet
import numpy as np
import RLSTM.Encoder as Encoder
import RLSTM.encoder_lstm as encoder_lstm
import RLSTM.Decoder as Decoder
import matplotlib.pyplot as plt
import RLSTM.Decoder_lstm as Decoder_lstm
import RLSTM.encoder_gru as encodet_gru
import RLSTM.encoder_rnn as encoder_rnn
import os
import datetime
tf.reset_default_graph()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
tf.reset_default_graph()

class parameter(object):
    def __init__(self):
        '''
        used to set the batch_size
        '''
        self.batch_size=64
        self.is_training=True
        self.encoder_layer=1
        self.decoder_layer=1
        self.encoder_nodes=128
        self.prediction_size=24
        self.learning_rate=0.001
        self.time_size=72
        self.features=15

'''
para used to set the parameters in later process
'''

class train(object):
    def __init__(self,time_size,features,prediction_size):
        self.x_input=tf.placeholder(dtype=tf.float32,shape=[None,time_size,features],name='pollutant')
        self.y=tf.placeholder(dtype=tf.float32,shape=[None,prediction_size])
    def trains(self,batch_size,encoder_layer,decoder_layer,encoder_nodes,prediction_size,is_training):
        '''

        :param batch_size: 64
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: True
        :return:
        '''
        # #this step use to encoding the input series data
        encoder_init=Encoder.encoder(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        (c_state, h_state)=encoder_init.encoding()

        #encoder_init=encoder_lstm.lstm(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        #encoder_init=encodet_gru.gru(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        # encoder_init=encoder_rnn.rnn(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        # h_state=encoder_init.encoding()

        #this step to presict the polutant concentration
        decoder_init=Decoder_lstm.lstm(batch_size,prediction_size,decoder_layer,encoder_nodes,is_training)
        pre=decoder_init.decoding(h_state)

        self.cross_entropy = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(self.y - pre), axis=0)), axis=0)
        # backprocess and update the parameters
        # self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)
        # return self.cross_entropy,self.train_op
    def test(self,batch_size,encoder_layer,decoder_layer,encoder_nodes,prediction_size,is_training):
        '''

        :param batch_size: usually use 1
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: False
        :return:
        '''
        encoder_init=Encoder.encoder(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        (c_state, h_state)=encoder_init.encoding()

        # encoder_init = encoder_lstm.lstm(self.x_input, batch_size, encoder_layer, encoder_nodes, is_training)
        # encoder_init = encodet_gru.gru(self.x_input, batch_size, encoder_layer, encoder_nodes, is_training)
        # encoder_init = encoder_rnn.rnn(self.x_input, batch_size, encoder_layer, encoder_nodes, is_training)
        # h_state = encoder_init.encoding()

        #this step to presict the polutant concentration
        decoder_init=Decoder_lstm.lstm(batch_size,prediction_size,decoder_layer,encoder_nodes,is_training)
        self.pre=decoder_init.decoding(h_state)
        return self.pre
    def accuracy(self,Label,Predict,epoch,steps):
        '''

        :param Label: represents the observed value
        :param Predict: represents the predicted value
        :param epoch:
        :param steps:
        :return:
        '''
        error = Label - Predict
        average_Error = np.mean(np.fabs(error.astype(float)))
        print("After %d epochs and %d steps, MAE error is : %f" % (epoch, steps, average_Error))

        RMSE_Error = np.sqrt(np.mean(np.square(np.array(Label) - np.array(Predict))))
        print("After %d epochs and %d steps, RMSE error is : %f" % (epoch, steps, RMSE_Error))

        cor = np.mean(np.multiply((Label - np.mean(Label)),
                                  (Predict - np.mean(Predict)))) / (np.std(Predict) * np.std(Label))
        print('The correlation coefficient is: %f' % (cor))
        return average_Error,RMSE_Error,cor
    def describe(self,Label,Predict,epoch,prediction_size):
        if epoch == 10 or epoch == 30 or epoch == 50 or epoch == 70 or epoch == 90 or epoch == 100:
            plt.figure()
            # Label is observed value,Blue
            plt.plot(Label[24:48], 'b*:', label=u'actual value')
            # Predict is predicted value，Red
            plt.plot(Predict[24:48], 'r*:', label=u'predicted value')
            # use the legend
            # plt.legend()
            plt.xlabel("Time(hours)", fontsize=17)
            plt.ylabel("PM2.5(ug/m3)", fontsize=17)
            plt.title("The prediction of PM2.5  (epochs =" + str(epoch) + ")", fontsize=17)
            plt.show()

def begin():
    '''
    from now on,the model begin to training, until the epoch to 100
    '''
    para = parameter()
    training = train(para.time_size,para.features,para.prediction_size)
    pre=training.test(1,para.encoder_layer,para.decoder_layer,para.encoder_nodes,para.prediction_size,False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # model_file = tf.train.latest_checkpoint('ckpt/')
        # saver.restore(sess, model_file)
        for i in range(40,100):
            saver.restore(sess, 'ckpt/pollutant.ckpt-'+str(i))
            para.batch_size = 1
            para.is_training=False
            # reading for the test sets
            Label = list()
            Predict = list()
            for x, label in testSet.train_data(para.batch_size, para.time_size, para.prediction_size):
                s = sess.run((pre),feed_dict={training.x_input: x})
                Label.append(label)
                Predict.append(s)
            Label = np.reshape(np.array(Label), [1, -1])[0]
            Predict = np.reshape(np.array(Predict), [1, -1])[0]

            print(list(Predict[25:48]))
            print(list(Predict[100:124]))
            print(list(Predict[200:224]))
            print(list(Predict[300:324]))
            print(list(Predict[400:424]))
            print(list(Predict[500:524]))

            average_Error, RMSE_Error, cor=training.accuracy(Label,Predict,3,0)
            print('the average is %f:'%average_Error)
            print('the RMSE value is %f:'%RMSE_Error)
            print('the correlations value is %f:'%cor)
            training.describe(Label,Predict,100,para.prediction_size)
def main(argv=None):
    begin()

if __name__ == '__main__':
    main()