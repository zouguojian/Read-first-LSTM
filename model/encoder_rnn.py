import tensorflow as tf
class rnn(object):
    def __init__(self,input,batch_size,layer_num=1,nodes=128,is_training=True):
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
        self.input=input

    def encoding(self):
        if self.is_training == True:init_state = tf.zeros([self.batch_size, self.nodes])
        else:init_state = tf.zeros([self.batch_size, self.nodes])

        def rnn_cell(inputs, state):
            # concat the input and the state
            state = tf.concat([inputs, state], 1)
            with tf.variable_scope('rnn_layer', reuse=tf.AUTO_REUSE):
                weight = tf.get_variable("weights", [self.nodes + inputs.shape[-1], self.nodes],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
                bias = tf.get_variable('biases', [self.nodes],
                                       initializer=tf.constant_initializer(0.1))
            return tf.nn.tanh(tf.matmul(state, weight) + bias)

        # out put the store data
        out_put = []
        state = init_state
        for i in range(self.input.shape[1]):
            state = rnn_cell(self.input[:, i, :], state)
            out_put.append(state)
        # output the final state
        final_state = out_put[-1]
        return tf.convert_to_tensor(final_state)