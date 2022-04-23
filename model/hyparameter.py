# -- coding: utf-8 --

import argparse
class parameter(object):
    def __init__(self,parser):
        self.parser=parser

        self.parser.add_argument('--save_path', type=str, default='weights/RLSTM/', help='save path')
        self.parser.add_argument('--model_name', type=str, default='RLSTM', help='training or testing model name')

        self.parser.add_argument('--divide_ratio', type=float, default=0.8, help='data_divide')
        self.parser.add_argument('--is_training', type=bool, default=True, help='is training')
        self.parser.add_argument('--epoch', type=int, default=100, help='epoch')
        self.parser.add_argument('--step', type=int, default=1, help='step')
        self.parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.3, help='drop out')
        self.parser.add_argument('--features', type=int, default=15, help='numbers of the feature')
        self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')
        self.parser.add_argument('--input_length', type=int, default=6, help='input length')
        self.parser.add_argument('--output_length', type=int, default=6, help='output length')
        self.parser.add_argument('--predict_length', type=int, default=6, help='predict length')

        self.parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
        self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layer')

        self.parser.add_argument('--training_set_rate', type=float, default=0.7, help='training set rate')
        self.parser.add_argument('--validate_set_rate', type=float, default=0.15, help='validate set rate')
        self.parser.add_argument('--test_set_rate', type=float, default=0.15, help='test set rate')

        self.parser.add_argument('--file_train', type=str, default='data/train.csv',help='training set address')
        self.parser.add_argument('--file_val', type=str, default='data/val.csv', help='validation set address')
        self.parser.add_argument('--file_test', type=str, default='data/test.csv', help='test set address')
        self.parser.add_argument('--file_sp', type=str, default='data/sp.csv', help='sp set address')
        self.parser.add_argument('--file_dis', type=str, default='data/dis.csv', help='dis set address')
        self.parser.add_argument('--file_in_deg', type=str, default='data/in_deg.csv', help='in_deg set address')
        self.parser.add_argument('--file_out_deg', type=str, default='data/out_deg.csv', help='out_deg set address')

        self.parser.add_argument('--file_out', type=str, default='ckpt', help='file out')

    def get_para(self):
        return self.parser.parse_args()

if __name__=='__main__':
    para=parameter(argparse.ArgumentParser())

    print(para.get_para().batch_size)