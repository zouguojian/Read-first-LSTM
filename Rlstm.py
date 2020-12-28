import tensorflow as tf
class rlstm(object):
    def __init__(self,layer_num=1,nodes=128,is_training=True):
        self.layer_num=layer_num    #the numbers of layer
        self.nodes=nodes            #the numbers of nodes to each layer
        self.is_training=is_training
    #the funtion of init_state() is used to initialized the state of C and H
    def train_state(self,batch_size):
        '''
        it divided train or test
        :return:
        '''
        with tf.variable_scope(name_or_scope='train_state', reuse=tf.AUTO_REUSE):
            c_state=tf.Variable(tf.zeros(shape=[batch_size,self.nodes],dtype=tf.float32),name='c_state')
            h_state=tf.Variable(tf.zeros(shape=[batch_size,self.nodes],dtype=tf.float32),name='h_state')
            return c_state,h_state

    def test_state(self,batch_size):
        with tf.variable_scope(name_or_scope='test_state', reuse=tf.AUTO_REUSE):
            c_state = tf.Variable(tf.zeros(shape=[batch_size, self.nodes], dtype=tf.float32),name='c_state')
            h_state = tf.Variable(tf.zeros(shape=[batch_size, self.nodes], dtype=tf.float32),name='h_state')
            return c_state, h_state
    def lstm_layer(self,inputs,c_state, h_state):
        '''

        :param input:
        :return:
        '''
        input = tf.concat([inputs, c_state, h_state], axis=1)
        #the fist gate, read gate
        #output shape is [batch_size,h+x+c]

        read_gate=tf.layers.dense(inputs=input,units=input.shape[1],
                                  activation=tf.nn.sigmoid,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  bias_initializer=tf.constant_initializer(0.0),name='read_gate')
        '''
        #contral data is that the read gate to control data stream,
        #read the important information in the inputs.
        '''
        control_data=tf.multiply(read_gate,input)

        '''
        #forget gate, it is means that use the read gate output 
        and the model inputs to forget the un useless data .
        input size is[batch_size,h+x+c], the output size is [batch_size,nodes]
        '''

        forget_gate=tf.layers.dense(inputs=control_data,units=self.nodes,
                                  activation=tf.nn.sigmoid,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  bias_initializer=tf.constant_initializer(0.0),name='forget_gate')

        #write opration is used to update the cell state
        #the output size is :[batch_size,nodes]
        write=tf.layers.dense(inputs=control_data,units=self.nodes,
                                  activation=tf.nn.sigmoid,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  bias_initializer=tf.constant_initializer(0.0),name='write')
        '''
        it used to update the cell state,C
        and the shape is [batch_size,nodes]
        '''
        C_hat=tf.layers.dense(inputs=control_data,units=self.nodes,
                                  activation=tf.nn.tanh,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  bias_initializer=tf.constant_initializer(0.0),name='c_hate')

        '''
        the final cell state C, and used to reference as the O(output)
        the size is [batch_size,nodes] 
        '''

        self.C=tf.multiply(forget_gate,c_state)+tf.multiply(write,C_hat)

        '''
        We can used this formula to achieve the traditional lSTM output, it combine C_hat and control_data
        the size of out is[batch_size,nodes]
        '''
        # O=tf.layers.dense(inputs=control_data,units=self.nodes,
        #                           activation=tf.nn.tanh,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                           bias_initializer=tf.constant_initializer(0.0),name='out')
        # self.H=tf.multiply(O,tf.tanh(self.C))
        #
        self.H=tf.tanh(self.C)

        return (self.H,self.C)

    def calculate(self,input,batch_size):
        '''
        This function
        :param input: input,and the size is [bacth_size,time_size,data_features]
        :return:
        '''


        self.c_states=[]
        self.h_states=[]

        '''
        use two layer loop ,the first loop to divided the layer,and the second layer loop 
        used to extract the time series features
        '''
        for layer in range(self.layer_num):
            with tf.variable_scope(name_or_scope=str(layer),reuse=tf.AUTO_REUSE):
                if self.is_training:c_state,h_state=self.train_state(batch_size)
                else:c_state,h_state=self.test_state(batch_size)
                # c_state, h_state = self.test_state(batch_size)
                h = []
                for time in range(input.shape[1]):
                    (c_state,h_state)=self.lstm_layer(input[:,time,:],c_state,h_state)
                    #store the each time h_state as the next layer input
                    h.append(h_state)
                #reshape as [batch_size,time_size,self.nodes]
                input=tf.reshape(tf.convert_to_tensor(h,dtype=tf.float32),shape=(-1,input.shape[1],self.nodes))
                #the state of each layer, the end time
                self.c_states.append(c_state)
                self.h_states.append(h_state)
        return (self.c_states[-1],self.h_states[-1])


# x=tf.Variable(tf.constant([[[2,3,4,5],[2,3,4,5]]],dtype=tf.float32))
# print(x.shape)
# lstm=rlstm(1,2,256)
# result=lstm.calculate(x)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(result).shape)