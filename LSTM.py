import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import copy
import requests
import io

# 加载数据
df = pd.read_excel('数据.xlsx')
data = np.transpose((np.array(df['风能产电量'])))
# normalize
normalize_data = (data - np.mean(data)) / np.std(data);
k=normalize_data[1:3];
normalized_data=normalize_data[0:len(normalize_data)-365];
test_data=normalize_data[-367:-1];
train_x, train_y = [], [];
seq_size = 3##time_step
# 序列大小为3

rnn_unit = 10  # hidden layer units
batch_size = 12  # 每一批次训练多少个样例
input_size = 1  # 输入层维度
output_size = 1  # 输出层维度
lr = 0.0006  # 学习率
test_length = 10;
tf.reset_default_graph();
for i in range(len(normalized_data) - seq_size - 1):
    train_x.append(np.expand_dims(normalized_data[i: i + seq_size], axis=1).tolist())
    train_y.append(normalized_data[i + 1: i + seq_size + 1].tolist())
X = tf.placeholder(tf.float32, [None, seq_size, input_size])
Y = tf.placeholder(tf.float32, [None, seq_size, output_size])
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# regression
def ass_rnn(batch):
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        w_in = weights['in']
        b_in = biases['in']
        input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, seq_size, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, state_is_tuple=True)
        init_state = cell.zero_state(batch, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                     dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
        w_out = weights['out']
        b_out = biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states


def main():
    global batch_size;
    pred,_=ass_rnn(batch_size);
    # loss = tf.reduce_mean(tf.square(out - Y))
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])));
    train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            step = 0
            start = 0
            end = start + batch_size
            while (end < len(train_x)):
                trainx=np.reshape(train_x[start:end],[batch_size,seq_size,1]);
                trainy=np.reshape(train_y[start:end],[batch_size,seq_size,1]);
                _, loss_ = sess.run([train_op, loss], feed_dict={X: trainx, Y: trainy})
                start += batch_size
                end = start + batch_size
                # 每10步保存一次参数
                if step % 10== 0:
                    print(i, step, loss_)
                    print("保存模型：", saver.save(sess, './stock.model'))
                step += 1
    pred1, _ = ass_rnn(1);
    # saver = tf.train.Saver(tf.global_variable())
    with tf.Session() as sess:
        saver.restore(sess, './stock.model')
        prev_seq = train_x[-1]  # 对列表进行倒序处理
        prev_seq = np.expand_dims(prev_seq, 0)
        predict = [];
        for i in range(24):
            next_seq = sess.run(pred1, feed_dict={X: prev_seq})
            predict.append(next_seq[0])
            # m=i+2;
            # t=copy.deepcopy(m)
            # print(np.shape(prev_seq[0:]),np.shape(next_seq[0]))
            # prev_seq=np.transpose(predict)
            # prev_seq=next_seq[0]
            # print(np.shape(prev_seq[0:]),np.shape(next_seq[0]))
            prev_seq = np.vstack((prev_seq[1:], (np.reshape(next_seq, [1, seq_size, 1]))))
            # prev_seq=np.array(prev_seq)
            # prev_seq = tf.squeeze(prev_seq,[1])
            # print(np.shape(prev_seq))
            # print(np.shape(np.resize(prev_seq,(t,1,1))));
            # print(np.size(np.resize(prev_seq,(t,1,1))))
            # prev_seq=np.resize(prev_seq,(t,1))
            # prev_seq.reshape(-1,1)
            # cc=np.shape(np.column_stack((prev_seq[0:],next_seq[0])))
            # t=np.reshape(prev_seq[0:],[1,seq_size,1])
            # k=np.reshape(next_seq[0],[1,seq_size,1])
            # prev_seq=[t[0],k[0]];
            # prev_seq = np.reshape(np.vstack((k[0],t[0])),[m,7,1]);
        # predict = np.reshape(predict, [-1])
        b = [str(i) for i in predict]
        fl = open('result8.txt', 'w');
        for i in b:
            fl.write(i)
            fl.write("\n")
        fl.close()
        # plt.figure()
        # plt.plot(list(range(len(predict))), test_data[0:len(predict)], color='b');
        # plt.plot(list(range(len())), predict, color='r');
        # plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
        # plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
        # true_data=normalized_data[:,-365:int(len(normalized_data))];
        # plt.plot(list(range(365)),true_data,color='b')
        # plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
        # #plt.plot(list(range(learn_length+1, learn_length+1 + len(predict))), predict, color='r')
        # plt.plot(list(range(365)),predict,color='r')
        #         plt.show()


                    # for step in range(10000):
        #     _,loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
        #     if step % 10 == 0:
        #         # 用测试数据评估loss
        #         print(step, loss_)
        # tf.reset_default_graph()
if __name__ == '__main__':
    main();


