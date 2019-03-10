##数据可视化
import tensorflow as tf
import numpy as np

##构建图
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
a=tf.get_variable("a",[],tf.float32,initializer=tf.random_normal_initializer())
b=tf.get_variable("b",[],tf.float32,initializer=tf.random_normal_initializer())
pred=tf.add(tf.multiply(x,a,name="mul_op"),b,name="add_op")

loss=tf.square(y-pred,name="loss")
opt=tf.train.GradientDescentOptimizer(0.01)
#计算梯度
grads_and_vars=opt.compute_gradients(loss)
train_op=opt.apply_gradients(grads_and_vars)

#收集训练数据
tf.summary.scalar("a",a)
tf.summary.scalar("b",b)
tf.summary.scalar("loss",loss[0])
merged_summery=tf.summary.merge_all()

#写入文件
summary_writer=tf.summary.FileWriter('./log_graph')
summary_writer.add_graph(tf.get_default_graph())
init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(500):
        train_x=np.random.randn(1)
        train_y=2*train_x+np.random.randn(1)*0.01+10
        _,summary=sess.run([train_op,merged_summery],feed_dict={x:train_x,y:train_y})
        summary_writer.add_summary(summary,step)
