#2018-7-9
#cnn 实现手写数字识别
# softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# sess = tf.InteractiveSession()

#定义用于初始化的两个函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#定义卷积和池化的函数
#卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入大小相同
#池化用简单传统的2x2大小的模板做max pooling，因此输出的长宽会变为输入的一半
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#定义占位符x和y_
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
x_image = tf.reshape(x, [-1,28,28,1])

#第一层卷积，卷积在每个5x5的patch中算出32个特征
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# (因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积，每个5x5的patch会得到64个特征
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

#把h_pool1和权值向量进行卷积，加上偏置值，再用relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#有1024个神经元的全连接层，此时图片大小为7*7
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

#把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#-1表示任意值
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#为了减少过拟合，在输出层之前加入dropout。用一个placeholder
# 代表一个神经元的输出在dropout中保持不变的概率。
#这样可以在训练过程中启用dropout，在测试过程中关闭dropout。
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#softmax输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#训练和评估模型
#用更加复杂的ADAM优化器来做梯度最速下降
# 在feed_dict中加入额外的参数keep_prob来控制dropout比例
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#训练和加载模型model.save\model.restore
model = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10000):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.7})

        acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.7})
        print(str(epoch)+"次迭代后,Testing Accuracy is：" + str(acc_test) + "\n")

    model.save(sess, 'net_cnn/my_net_cnn.ckpt')

    print("模型训练好了！")

    model.restore(sess, 'net_cnn/my_net_cnn.ckpt')
    acc_test = sess.run(accuracy,feed_dict={x:mnist.test.images , y_:mnist.test.labels, keep_prob:0.7})
    print("after training,Testing Accuracy is：" + str(acc_test)+ "\n")