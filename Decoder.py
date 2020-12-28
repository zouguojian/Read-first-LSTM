import RLSTM.Rlstm as lstm
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
            self.encoder_Lstm=lstm.rlstm(batch_size,layer_num,nodes,is_training)
    def decoding(self):
        '''
        we always use c_state as the input to decoder
        '''
        # (self.c_state, self.h_state) = self.encoder_Lstm.calculate(self.h_state)
        h_state=self.h_state
        h=[]
        for i in range(self.predict_time):
            h_state=tf.reshape(h_state,shape=(-1,1,self.nodes))
            (c_state, h_state)=self.encoder_Lstm.calculate(h_state)
            #we use the list h to recoder the out of decoder eatch time
            h.append(h_state)

        h=tf.convert_to_tensor(h,dtype=tf.float32)
        #LSTM的最后输出结果
        h_state = tf.reshape(h, [-1, self.nodes])

        #the full connect layer to output the end results
        with tf.variable_scope('Layer', reuse=tf.AUTO_REUSE):
            w = tf.get_variable("wight", [self.nodes, 1],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable("biases", [1],
                                   initializer=tf.constant_initializer(0))
            results = tf.matmul(h_state, w) + bias
        return tf.reshape(results, [-1, self.out_num])