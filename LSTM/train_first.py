import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#---超参数设置-----
 
rnn_unit = 10    #隐层数量
input_size = 1   #每个样本特征数量
output_size = 1  #输出大小
time_step = 20   #时间步数
batch_size = 40  #每次训练集数量
lr = 0.0001      #学习率



#----从文件导入数据------
excel_path = 'data.xlsx'
df2012 = pd.read_excel(excel_path, sheetname='2012')
data2012=df2012.iloc[:,2:3].values    #二维数据集  第一维度是样本特征，第二维度是样本序列
df2013 = pd.read_excel(excel_path, sheetname='2013')
data2013=df2013.iloc[:,2:3].values    #二维数据集  第一维度是样本特征，第二维度是样本序列
df2014 = pd.read_excel(excel_path, sheetname='2014')
data2014=df2014.iloc[:,2:3].values    #二维数据集  第一维度是样本特征，第二维度是样本序列
df2015 = pd.read_excel(excel_path, sheetname='2015')
data2015=df2015.iloc[:,2:3].values    #二维数据集  第一维度是样本特征，第二维度是样本序列
df2016 = pd.read_excel(excel_path, sheetname='2016')
data2016=df2016.iloc[:,2:3].values    #二维数据集  第一维度是样本特征，第二维度是样本序列
data=[]
data=np.vstack((data2012,data2013))
data=np.vstack((data,data2014))
data=np.vstack((data,data2015))
data=np.vstack((data,data2016))
data_len=len(data)

#-----获取训练集----
def get_train_data(batch_size=20,time_step=20,train_begin=0,train_end=100):
    batch_index=[]
    data_train=data[train_begin:train_end]          #debug
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step]
       y=normalized_train_data[i+1:i+time_step+1]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

#----定义神经网络变量----
#输入层、输出层权值、偏量
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }
#----定义lstm神经网络----
def lstm(X):      
    batch=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])   
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states
#----训练模型----
def train_lstm(batch_size=20,time_step=20,train_begin=0,train_end=100):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print("Number of iterations:",i," loss:",loss_)
        print("model_save: ",saver.save(sess,'model_save\modle.ckpt'))
        #我是在window下跑的，这个地址是存放模型的地方，模型参数文件名为modle.ckpt
        #在Linux下面用 'model_save2/modle.ckpt'
        print("The train has finished")
train_lstm(batch_size,time_step,0,data_len)

