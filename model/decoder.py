from model.rlstm import rlstm
import tensorflow as tf
class decoder(object):
    def __init__(self,h_state,batch_size,predict_time,layer_num=1,nodes=128,is_training=True):
        '''
        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        we need to define the decoder of Encoder-Decoder Model,and the parameter will be
        express in the Rlstm.
        '''
        self.h_state=h_state
        self.predict_time=predict_time
        self.nodes=nodes
        self.out_num=1

        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            self.encoder_rlstm = rlstm(batch_size,layer_num,nodes,is_training)

    def decoding(self):
        '''
        we always use c_state as the input to decoder
        '''
        # (self.c_state, self.h_state) = self.encoder_Lstm.calculate(self.h_state)
        h_state=self.h_state
        h=[]
        for i in range(self.predict_time):
            h_state=tf.reshape(h_state,shape=(-1,1,self.nodes))
            (c_state, h_state)=self.encoder_rlstm.calculate(h_state)
            #we use the list h to recoder the out of decoder eatch time
            h.append(h_state)

        h = tf.convert_to_tensor(h,dtype=tf.float32)
        h_state = tf.reshape(h, [-1, self.nodes])
        results = tf.layers.dense(inputs=h_state, units=1, name='pre_y', reuse=tf.AUTO_REUSE)
        return tf.reshape(results, [-1, self.out_num])