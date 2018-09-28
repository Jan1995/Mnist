import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets ( "MNIST_data/" , one_hot=True )
def init(shape,s):
	'''
	随机初始化函数
	:param s: 需要初始化的变量
	:param shape: 输出张量形状
	:return: tf.Variable
	'''
	if s=='w':return tf.Variable(tf.random_normal(shape,stddev=0.1))
	elif s=='b':return tf.Variable(tf.zeros(shape))
	else:print('参数不合法')
def conv(x,w,padding):
	'''
	卷积函数
	:param x: 输入
	:param w: 卷积核
	:param padding: 模式
	:return: 卷积结果
	'''
	return tf.nn.conv2d(x,w,[1,1,1,1],padding)
def pool(x):
	'''
	池化函数
	:param x: 输入
	:return: 用[1,2,2,1]移动，maxpool结果
	'''
	return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],'SAME')
def cost(y,y_):
	'''
	计算交叉熵
	:param y: lable
	:param y_: 训练结果
	:return:
	'''
	return -tf.reduce_sum(y*tf.log(y_))

#输入图片和标记占位符
x0=tf.placeholder('float',[None,784])
y=tf.placeholder('float',[None,10])

#初始化
x=tf.reshape(x0,[-1,28,28,1])#对图片二维化
con_w1=init([5,5,1,6],'w')#第一层卷积核

b1=init([6],'b')#第一层卷积偏置
bs1=init([6],'b')#第一层池化偏置

con_w21=init([6,5,5,3,1],'w')#第二层对相邻3个通道卷积核  6对应range(6)
con_w22=init([6,5,5,4,1],'w')#第二层对相邻4个通道卷积核
con_w23=init([3,5,5,4,1],'w')#第二层对不相邻4个通道据卷积核
con_w24=init([5,5,6,1],'w')#第二层对所有通道卷积核

b2=init([16],'b')#第二层卷积偏置
bs2=init([16],'b')#第二层池化偏置

con_w3=init([5,5,16,120],'w')#第三层卷积核

b3=init([120],'b')#第三层卷积偏置

fw=init([5*5*120,84],'w')#全连接层权重
fb=init([84],'b')#全连接层偏置

ow=init([84,10],'w')#softmax层权重
ob=init([10],'b')#softmax层偏置

#构建模型
c1=conv(x,con_w1,'SAME')+b1#c1

s2=tf.nn.tanh(pool(c1)+bs1)#s2

c30=tf.concat([s2,s2],3)
c30=tf.split(c30,12,3)
c31=tf.concat([conv(tf.concat(c30[i:i+3],3),con_w21[i],'VALID')for i in range(6)],3)
c32=tf.concat([conv(tf.concat(c30[i:i+4],3),con_w22[i],'VALID')for i in range(6)],3)
c33=tf.concat([conv(tf.concat([tf.concat(c30[i:i+2],3),tf.concat(c30[i+3:i+5],3)],3),con_w22[i],'VALID')for i in range(3)],3)
c34=conv(s2,con_w24,'VALID')
c3=tf.concat([c31,c32,c33,c34],3)#c3
s4=tf.nn.tanh(pool(c3)+bs2)#s4

c5=conv(s4,con_w3,'SAME')+b3#c5

f6=tf.matmul(tf.reshape(c5,[-1,5*5*120]),fw)+fb#全连接
output=tf.nn.softmax(tf.matmul(f6,ow)+ob)#softmax
train=tf.train.AdamOptimizer(0.001).minimize(cost(y,output))#adam训练


with tf.Session() as sess:
	sess.as_default()
	sess.run(tf.initialize_all_variables())
	for epoch in range(51):
		batch_x,batch_y=mnist.train.next_batch(100)#取训练集中一批，100个数据进行训练
		sess.run(train,feed_dict={x0:batch_x,y:batch_y})
		if not epoch%10:
			testbatch_x,testbatch_y=mnist.test.next_batch(1000)
			y_=sess.run(output,feed_dict={x0:testbatch_x})
			y_=np.array(np.argmax(y_,1))

			# result = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
			result=np.mean(np.equal(np.argmax(testbatch_y,1),y_))
			print('第',str(epoch),'次训练，正确率为',result*100,'%')

    model.save(sess, 'net/my_net_cnn.ckpt')

    print("模型训练好了！")