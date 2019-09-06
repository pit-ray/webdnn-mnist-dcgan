#coding: utf-8
import os
from PIL import Image
import cupy as cp
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from utilities import label2onehot


class Generator(chainer.Chain):
    def __init__(self, z_dim):
        super().__init__()
        class_num = 10 #0-9
        self.img_shape = (1, 28, 28)

        he_w = chainer.initializers.HeNormal()
        xavier_w = chainer.initializers.LeCunNormal()
        ch_dim = 128
        with self.init_scope():
            self.l1 = L.Linear(z_dim + class_num, ch_dim*3*3, initialW=he_w)
            self.dc1 = L.Deconvolution2D(ch_dim, ch_dim // 2, ksize=3, stride=2, initialW=he_w)
            self.dc1_bn = L.BatchNormalization(size=ch_dim // 2)
            self.dc2 = L.Deconvolution2D(ch_dim // 2, ch_dim // 4, ksize=2, stride=2, initialW=he_w)
            self.dc2_bn = L.BatchNormalization(size=ch_dim // 4)
            self.dc3 = L.Deconvolution2D(ch_dim // 4, 1, ksize=2, stride=2, initialW=xavier_w)

        self.z_dim = z_dim
        self.class_num = class_num
        self.img_counter = 0

    def forward(self, x):
        batch_size = x.shape[0]

        #Project and reshape
        h1 = F.reshape(F.relu(self.l1(x)), (batch_size, 128, 3, 3))

        #conv1
        h2 = F.relu(self.dc1_bn(self.dc1(h1)))

        #conv2
        h3 = F.relu(self.dc2_bn(self.dc2(h2)))

        #conv3 (none batch norm)
        h4 = self.dc3(h3)
        #h4 = F.tanh(h4)
        h4 = F.sigmoid(h4)
        self.out = h4 #to save img

        return h4

    def save_img(self, out_dir='img', device=0, row=10, col=10):
        os.makedirs(out_dir, exist_ok=True)
        img_data = self.out.array

        if device != -1:
            img_data = cp.asnumpy(img_data)

        batch_size, ch_size, h_size, w_size = img_data.shape
        img_data = np.reshape(img_data, (batch_size, h_size, w_size))
        tile_img = None
        for i in range(row):
            most_left = i * col
            row = img_data[most_left]
            for j in range(1, col):
                row = np.concatenate((row, img_data[most_left + j]), axis=1)

            if i == 0:
                tile_img = row
            else:
                tile_img = np.concatenate((tile_img, row), axis=0)

        img_array = np.uint8(tile_img*255)
        img = Image.fromarray(img_array)
        img.save(out_dir + '/tile_img-' + str(self.img_counter) + '.png')

        self.img_counter += 1

    def generate_noise(self, device, batch_size, is_random=False):
        #select module
        xp = cp
        if device == -1:
            xp = np
        noise_z = xp.random.normal(size=(batch_size, self.z_dim)).astype(np.float32)

        if not is_random:
            #fix number
            num = xp.arange(self.class_num).astype(np.int32)
            fake_label = xp.tile(num, int(batch_size / self.class_num))
            fake_label = xp.concatenate((fake_label, num[:batch_size % self.class_num]))
        else:
            #noise number
            fake_label = xp.random.randint(0, self.class_num, size=batch_size)

        #Conditional Z
        noise_y = label2onehot(fake_label, self.class_num)
        noise_zy = xp.concatenate((noise_z, noise_y), axis=1)
        noise_zy = chainer.Variable(noise_zy)

        return (noise_zy, fake_label)

class Discriminator(chainer.Chain):
    def __init__(self):
        super().__init__()
        class_num = 10 #0-9
        he_w = chainer.initializers.HeNormal()
        xavier_w = chainer.initializers.LeCunNormal()
        ch_dim = 32

        with self.init_scope():
            self.c1 = L.Convolution2D(1 + class_num, ch_dim, ksize=2, stride=2, initialW=he_w) #concatenate class info to channel axis of img
            self.c2 = L.Convolution2D(ch_dim, ch_dim*2, ksize=2, stride=2, initialW=he_w)
            self.c2_bn = L.BatchNormalization(size=ch_dim*2)
            self.c3 = L.Convolution2D(ch_dim*2, ch_dim*4, ksize=3, stride=2, initialW=he_w)
            self.c3_bn = L.BatchNormalization(size=ch_dim*4)
            self.l1 = L.Linear((ch_dim*4)*3*3, 1, initialW=xavier_w)

        self.class_num = class_num

    def forward(self, x):
        #conv1 (none batch norm)
        h1 = F.leaky_relu(self.c1(x))

        #conv2
        h2 = self.c2(h1)
        #h2 = self.c2_bn(h2)
        h2 = F.leaky_relu(h2)

        #conv3
        h3 = self.c3(h2)
        #h3 = self.c3_bn(h3)
        h3 = F.leaky_relu(h3)

        #resize to unit one
        h4 = self.l1(h3)
        #DCGAN
        h4 = F.sigmoid(h4)

        return h4
