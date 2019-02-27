# coding = gbk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys


rnn_unit=20       #hidden layer units
input_size=9
#input_size=2
output_size=1
lr=0.001      
th = 0
n_train = 1750   


#f=open('data5f.csv') 
df=pd.read_csv('app.csv')
#df=pd.read_csv('test.csv')
if df.empty:
   print("Data is empty")
data=df.iloc[0:,0:].values
checkpoint_dir = ''
print(data)
print(type(data))


def get_train_data(batch_size=100,time_step=20,train_begin=0,train_end=5112):
    batch_index=[]
    data_train=data[0:,0:]
    #print(data_train)
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  
    #print(normalized_train_data)
    train_x=[]

    for i in range(len(normalized_train_data)-20):
       #print(i)
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,0:]
      # y=data[i:i+time_step,0,np.newaxis]
       train_x.append(x.tolist())
      # train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-20))
    #print (batch_index)
    #print(train_x)
    #print(train_y)
    print("finish1")
    return batch_index,train_x



def get_test_data(time_step=20):
    print("enter test")
    data_test=data[0:,0:]
    print(data_test)
    mean=np.mean(data_test,axis=0)                        #列平均值
    std=np.std(data_test,axis=0)                          #列标准差
    normalized_test_data=(data_test-mean)/std  
    print(normalized_test_data)
    size=(len(normalized_test_data)+time_step-1)//time_step
    test_x=[]
    i=0
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,0:]
       #y=data[i*time_step:(i+1)*time_step,0]
       test_x.append(x.tolist())
      # test_y.extend(y)
       #i = i+1


    test_x.append((normalized_test_data[(i+1)*time_step:,0:]).tolist())
   # test_y.extend((data[(i+1)*time_step:,0]).tolist())
    
    print("finish test")
    return mean,std,test_x





weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

 
#定义神经网络
def lstm_1(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    print("shape secc1")
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    with tf.variable_scope('lstm') as lstm1:
          output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype = tf.float32)  
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

def lstm_2(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    print("shape secc1")
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size]) 
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit]) 
    
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    with tf.variable_scope('lstm', reuse=True) as lstm2:
          output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype = tf.float32)  
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    #print("********")
    return pred,final_states

def train_lstm(batch_size=100,time_step=20,train_begin=0,train_end=5112):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    print("enter train")
    batch_index,train_x=get_train_data(batch_size,time_step,train_begin,train_end)
    print("get train data")
    pred,_=lstm_1(X)
    print("finish model")
    loss1 = 0

    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss) #最小化方差
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint('')    
    print("enter train")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)

        #i=0
        for i in range(1):
            #for step in range(len(batch_index)-1):
             #   _,loss1 =sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]]})

           # print(i,loss1)
            if i % 1==0:
                print("save model: ",saver.save(sess,'D:\JavaProgram\LogAnalysis\App\stock2.model',global_step=i))
    print("finish train!")


train_lstm()



def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x=get_test_data(time_step)
    #print(test_x)
    pred,_=lstm_2(X)     
    saver=tf.train.Saver(tf.global_variables())
   
    result = []
    probability=[]
    sum_acc = 0
    acc_error = 0
    test_error = 0
    true_err = 0
    sum_miss = 0
    error = 0
    
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('')
        #print("********")
        saver.restore(sess, module_file) 
        
        test_predict=[]
        print("acc")
        i=0
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})   
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        print("acc finish")
        i=0
        max_pre = max(test_predict)
        while(i<len(test_predict)):
            test_predict[i] = (test_predict[i] + max_pre)/(2*max_pre)
            i=i+1
        i=0
        #print(len(test_predict))
        pre0=str(test_predict)
        pre0=pre0.replace("[","")
        pre0=pre0.replace(",","\n")
        pre0=pre0.replace("]","")+"\n"
        f=open('test_prediction.txt','w')
        f.write(pre0)
        f.close()
        while(i<len(test_predict)):
            probability.append(test_predict[i])
            if(test_predict[i] < th):
                result.append(1)
            else:
                result.append(0)
            i=i+1
        pre=str(result)
        pre=pre.replace("[","")
        pre=pre.replace(",","\n")
        pre=pre.replace("]","")+"\n"
        f=open('demo.txt','w')
        f.write(pre)
        f.close()

        #pre2=str(probability)
        #pre2=pre2.replace("[","")
        #pre2=pre2.replace(",","\n")
        #pre2=pre2.replace("]","")+"\n"
        #f2=open('pra.txt','w')
        #f2.write(pre2)
        #f2.close()

        
        i=0
        while(i<len(test_predict)):
            if(result[i]==0):
                test_error = test_error+1
            i=i+1
        
 
        print("ture error = %d"%true_err)
        print("test error = %d"%test_error)
        print("test exact error = %d"%acc_error)
        print("miss report = %d"%sum_miss)
        print("error report = %d"%error)
       
        #plt.figure()
        #plt.plot(list(range(len(test_predict))), test_predict, color='b')
        #i=0
        #while(i < len(test_y)):
            #if(test_y[i]==0):
             #    plt.axhline(y=test_predict[i],  color='r', linewidth = 0.5)
                 
            #i = i+1
       # plt.plot(list(range(len(test_y))), test_y,  color='r')
        print("end predict!")
        #plt.show()
       
prediction() 