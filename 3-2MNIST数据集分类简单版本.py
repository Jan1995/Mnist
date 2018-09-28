import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

model = tf.train.Saver()#训练和加载模型model.save\model.restore

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    model.restore(sess,'net/my_net.ckpt')
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
    #
    # model.save(sess,'net/my_net.ckpt')

        acc_train1 = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        print("第" + str(epoch) + "次迭代后,***Trainning Accuracy is：" + str(acc_train1))
        #
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
        # acc_train = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        # print("第" + str(epoch) + "次迭代后,Trainning Accuracy is：" + str(acc_train))
        #
        # acc_test = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        # print("第" + str(epoch) + "次迭代后,Testing Accuracy is：" + str(acc_test)+ "\n")






