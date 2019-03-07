##数据可视化
import tensorflow as tf
import numpy as np

##构建图
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
a=tf.get_variable("a",[],tf.float32,initializer=tf.random_normal_initializer())
b=tf.get_variable("b",[],tf.floar32,initializer=tf.random_mormal_initializer())
pred=tf.add(tf.multiply(x,a,name="mul_op"),b,name="add_op")

loss=tf.square(y-pred,name="loss")
opt=tf.train.GradientDescentOptimizer(0.01)
#计算梯度
grads_and_vars=opt.compute_gradients(loss)
tarin_op=opt.apply_gradients(grads_and_vars)

#收集训练数据
tf.summary.scalar("a",a)
tf.summary.scalar("b",b)
tf.summary.scalar("loss",loss)
merged_summery=tf.summary.merge_all()
