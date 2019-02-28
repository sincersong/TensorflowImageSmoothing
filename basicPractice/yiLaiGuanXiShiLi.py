
##在tensorflow的图生成以后，运行时并不会全部运行，只会运行相关结果的依赖部分
##如下面的例子

import tensorflow as tf
import tensorflow as tf2

x=tf.Variable(0.0,name="x")
x_plus_1=tf.assign_add(x,1,name="x_plus")     #x_plus_1依赖于x

with tf.control_dependencies([x_plus_1]):     #with操作中的y直接等于x，即y依赖于x
    y=x

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(5):                        #运行中只执行相关依赖，所以不执行x_plus_1的操作
        print(y.eval())                       #所以结果均为0



x2=tf2.Variable(0.0,name="x2")
x2_plus_1=tf2.assign_add(x,1,name="x2_plus")

with tf2.control_dependencies([x2_plus_1]):     #with中添加y2的新操作，使得y2依赖于x2_plus_1
    y2=tf2.identity(x,name="y2")

init=tf2.global_variables_initializer()
with tf2.Session() as sess2:
    sess2.run(init)
    for i in range(5):                        #运行中只执行相关依赖，所以执行x_plus_1的操作
        print(y2.eval())                       #所以结果为1,2,3,4,5
