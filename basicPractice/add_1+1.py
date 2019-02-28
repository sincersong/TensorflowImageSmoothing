import tensorflow as tf

##创建图
v1=tf.constant(1,name='value1')
v2=tf.constant(1,name='value2')
add_op=tf.add(v1,v2,name='add_op_name')

##用with结构在会话中执行操作
with tf.Session() as sess:
    ##执行运算，并将结果赋给python变量
    result=sess.run(add_op)
    print('1+1=%.0f'%result)
