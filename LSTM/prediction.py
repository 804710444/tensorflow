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
df2016 = pd.read_excel(excel_path, sheetname='2016')
data2016=df2016.iloc[:,2:3].values    #二维数据集  第一维度是样本特征，第二维度是样本序列
df2017 = pd.read_excel(excel_path, sheetname='2017')
data2017=df2017.iloc[:,2:3].values    #二维数据集  第一维度是样本特征，第二维度是样本序列
data=[]
data=np.vstack((data2016,data2017))
data_len=len(data)



#-----获取测试集----
def get_test_data(time_step=20,test_begin=0,test_end=100):
    data_test=data[test_begin:test_end]     
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    test_x,test_y=[],[]   #训练集
    for i in range(len(normalized_test_data)-time_step):
       x=normalized_test_data[i:i+time_step]
       y=normalized_test_data[i+time_step]
       test_x.append(x.tolist())
       test_y.extend(y.tolist())
    return mean,std,test_x,test_y

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

#————————————————预测模型————————————————————
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step,140,200)
    
    with tf.variable_scope("sec_lstm"):   
        pred,_=lstm(X)
   
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          test_predict.extend(prob[-1])

        test_y=np.array(test_y)*std[0]+mean[0]
        test_predict=np.array(test_predict)*std[0]+mean[0]
        
        
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差程度
        print("The accuracy of this predict:",acc)
        #以折线图表示结果
        for i in range(len(test_predict)):
            if test_predict[i] <0 :
                test_predict[i] =1 
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

prediction()
