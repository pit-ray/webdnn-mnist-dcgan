#coding: utf-8
import numpy as np

import chainer

def label2onehot(label_batch, class_num):
    xp = chainer.cuda.get_array_module(label_batch)
    one_hot = xp.eye(class_num).astype(np.float32)

    buf = one_hot[label_batch]

    return buf

def label2images(label_batch, class_num, img_shape):
    xp = chainer.cuda.get_array_module(label_batch)
    one_hot = label2onehot(label_batch, class_num)
    row_size = img_shape[1]
    col_size = img_shape[2]

    onehot_img = xp.reshape(one_hot, (-1, class_num, 1, 1))
    onehot_img = xp.repeat(onehot_img, row_size, axis=2)
    onehot_img = xp.repeat(onehot_img, col_size, axis=3)

    return onehot_img