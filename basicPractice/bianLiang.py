import tensorflow as tf
import tensorflow as tf2

#设置变量
weight1=tf.Variable(0.01)
weight2=tf.Variable(weight1.initial_value*2)

##变量需要初始化（常量不需要） 此为初始化语句
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("The weight1 is")
    print(sess.run(weight1))
    print("The weight2 is")
    print(sess.run(weight2))

################################################################################
#变量一旦设置，其形式不变，除非变形
v1=tf2.Variable([1,2,3,4,5,6,7,8,9])
v2=tf2.reshape(v1,[3,3])

in=tf2.global_variables_initializer()
with tf2.Session() as se:
    se.run(in)
    print(se.run(v1))
    print(se.run(v2))
