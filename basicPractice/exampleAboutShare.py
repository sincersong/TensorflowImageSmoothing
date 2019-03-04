##采用共享变量的方式来训练数据
##本题以函数 y=3*x*x+5*x+8 为例来训练网络

import tensorflow as tf                ##导包
import numpy as np

def get_data(number):                 ##获取训练的数据库
    list_x=[]
    list_label=[]
    for i in range(number):
        x=np.random.randn(1)*10
        y=3*x*x+5*x+8+np.random.randn(1)
        list_x.append(x)
        list_label.append(y)
    return list_x,list_label

def inference(x):                    ##定义计算网络
    a=tf.Variable(0.01,name="a")
    b=tf.Variable(0.01,name="b")
    c=tf.Variable(0.01,name="c")
    y=a*x*x+b*x+c
    return y

train_x=tf.placeholder(tf.float32)         ##设置数据的占位变量
train_label=tf.placeholder(tf.float32)
test_x=tf.placeholder(tf.float32)
test_label=tf.placeholder(tf.float32)

with tf.variable_scope("inference"):       ##设置域内关系，并且设置域内共享前面的变量（名称必须相同）
    tf.get_variable_scope().reuse_variables()
    train_y=inference(train_x)
    test_y=inference(test_x)

train_loss=tf.square(train_label-train_y)   ##定义损失函数、训练迭代方法和训练目标
test_loss=tf.square(test_label-test_y)
opt=tf.train.GradientDescentOptimizer(0.001)
train_op=opt.minimize(train_loss)

init=tf.global_variables_initializer        ##初始化变量

train_data_x,train_data_label=get_data(1000)     ##读取数据
test_data_x,test_data_label=get_data(100)

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_op,feed_dict={train_x:train_data_x[i],train_label:train_data_label[i]})
        if i%10 ==0:
            test_loss_value=sess.run(test_loss,feed_dict={test_x:test_data_x[i%10],test_label:test_data_label[i%10]})
            print("step: %d  eval loss is %3f  "%(i,test_loss_value))
