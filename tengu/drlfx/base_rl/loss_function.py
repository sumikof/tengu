import tensorflow as tf
from keras import backend as k


# [1]損失関数の定義
# 損失関数にhuber関数を使用します 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = k.abs(err) < 1.0
    L2 = 0.5 * k.square(err)
    L1 = (k.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return k.mean(loss)
