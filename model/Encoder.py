from model.rlstm import rlstm
import tensorflow as tf
class encoder(object):
    def __init__(self,input,batch_size,layer_num=1,nodes=128,is_training=True):
        '''
        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        """We need to define the encoder of Encoder-Decoder Model,and the parameter will be
        express in the Rlstm."""

        encoder_rstm=rlstm(batch_size=batch_size,layer_num=layer_num, nodes=nodes, is_training=is_training)
        (self.c_state,self.h_state)=encoder_rstm.calculate(input,batch_size)

    def encoding(self):
        '''
        :return:
        '''
        """we always use c_state as the input to decoder"""

        return (self.c_state,self.h_state)