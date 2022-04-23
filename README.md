# Improved-RLSTM  

---

## 注意事项

<font face="微软雅黑" >需要注意的是，需要根据requirements.txt文件中指示的包进行安装，才能正常的运行程序！！！</font>
  
>* 首先，使用conda创建一个虚拟环境，如‘conda create prediction’；  
> * 激活环境，conda activate prediction；  
> * 安装环境，需要安装的环境已经添加在requirements.txt中，可以用conda安装，也可以使用pip安装，如：conda install tensorflow==1.12.0；  
> * 如果安装的是最新的tensorflow环境，也没问题，tensorflow的包按照以下方式进行导入即可：import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()；  
> * 点击 run_train.py文件即可运行代码。
> * 需要注意的是，我们在tensorflow的1.12和1.14版本环境中都可以运行
--- 
Used to extract the series time features than traditional lstm, if you count some problems, please describe the types of problems.  

In the training step: open your terminal - python run_train.py 1  
In the test step: python run_train.py 0  


Abstract—Accurate air pollutant prediction allows effective environment management to reduce the impact of pollution and prevent pollution incidents. Existing studies of air pollutant prediction are mostly interdisciplinary involving environmental science and computer science where the problem is formulated as time series prediction. A prevalent recent approach to time series prediction is the Encoder-Decoder model, which is based on recurrent neural networks (RNN) such as long short-term memory (LSTM), and great potential has been demonstrated. An LSTM network relies on various gate units, but in most existing studies the correlation between gate units is ignored. This correlation is important for establishing the relationship of the random variables in a time series as the stronger is this correlation, the stronger is the relationship between the random variables. In this paper we propose an improved LSTM, named Read-first LSTM or RLSTM for short, which is a more powerful temporal feature extractor than RNN, LSTM and Gated Recurrent Unit (GRU). RLSTM has some useful properties: (1) enables better store and remember capabilities in longer time series and (2) overcomes the problem of dependency between gate units. Since RLSTM is good at long term feature extraction, it is expected to perform well in time series prediction. Therefore, we use RLSTM as the Encoder and LSTM as the Decoder to build an Encoder-Decoder model (EDSModel)  for pollutant prediction in this paper. Our experimental results show, for 1 to 24 hours prediction, the proposed prediction model performed well with a root mean square error of 30.218. The effectiveness and superiority of RLSTM and the prediction model have been demonstrated. 
Keywords—encoder-decoder model, recurrent neural networks, long short term memory, air pollutant prediction, deep learning, numerical analysis


Reference format:  
[1] Zhang, B. ,  Zou, G. ,  Qin, D. ,  Lu, Y. , &  Wang, H. . (2021). A novel encoder-decoder model based on read-first lstm for air pollutant prediction. Science of The Total Environment, 765(3), 144507.

LaTex format:  
@article{2021A,
  title={A novel Encoder-Decoder model based on read-first LSTM for air pollutant prediction},
  author={ Zhang, B.  and  Zou, G.  and  Qin, D.  and  Lu, Y.  and  Wang, H. },
  journal={Science of The Total Environment},
  volume={765},
  number={3},
  pages={144507},
  year={2021},
}
