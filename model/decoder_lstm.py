import tensorflow as tf
class lstm(object):
    def __init__(self,batch_size,predict_time,layer_num=1,nodes=128,is_training=True):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.is_training=is_training
        self.keep_pro()
        self.predict_time=predict_time
        self.out_num=1

    def keep_pro(self):
        '''
        used to define the self.keepProb value
        :return:
        '''
        if self.is_training:self.keepProb=0.5
        else:self.keepProb=1.0

    def lstm_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
        # if confirg.KeepProb<1:
        return tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keepProb)

    def decoding(self,h_state):
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        print(h_state.shape)
        initial_state=mlstm_cell.zero_state(self.batch_size,tf.float32)
        h=[]

        for i in range(self.predict_time):
            h_state = tf.expand_dims(h_state,axis=1)
            # h_state = tf.reshape(h_state, shape=(self.batch_size, 1, self.nodes))
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                h_state, state = tf.nn.dynamic_rnn(cell=mlstm_cell, inputs=h_state,initial_state = initial_state,dtype=tf.float32)
                initial_state=state
            result = tf.layers.dense(inputs=tf.squeeze(h_state,axis=1), units=1, name='result', reuse=tf.AUTO_REUSE)
            h.append(result)

        return tf.squeeze(tf.transpose(tf.convert_to_tensor(h),[1,2,0]),axis=1,name='pre_y')