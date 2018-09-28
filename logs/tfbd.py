import tensorflow as tf

input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
input2 = tf.constant([3.0, 2.0, 1.0], name='input2')
output = tf.add_n([input1, input2], name='add')

init = tf.global_variables_initializer()

merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('log_test/', sess.graph)
    sess.run(init)
    print(sess.run(output))
writer.close()

"""
# TensorBoard 基础语法功能
# 记录标量的变化
tf.summary.scalar('var_name', var)

# 记录向量或者矩阵，tensor的数值分布变化。
tf.summary.histogram('vec_name', vec)

# 把所有的记录并把他们写到 log_dir 中
merged = tf.summary.merge_all()

# 保存位置
train_writer = tf.summary.FileWriter(log_dir + '/add_example', sess.graph)

运行完后，在命令行中输入
tensorboard - -logdir = log_dir_path(你保存到log路径)
"""
