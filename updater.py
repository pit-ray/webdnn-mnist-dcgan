#coding: utf-8
import cupy as cp
import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
from chainer.training.updaters import StandardUpdater

from utilities import label2images, label2onehot


class GANUpdater(StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.dis_ln = kwargs.pop('dis_ln')
        self.is_random = kwargs.pop('is_random')
        super().__init__(*args, **kwargs)

        self.gen_opt = self.get_optimizer('gen')
        self.dis_opt = self.get_optimizer('dis')
        self.gen = self.gen_opt.target
        self.dis = self.dis_opt.target
        self.class_num = self.gen.class_num
        self.z_dim = self.gen.z_dim
        self.img_shape = self.gen.img_shape
        self.batch_size = self.get_iterator('main').batch_size

    def discriminator_loss(self, real_y, fake_y):
        L1 = -F.sum(F.log(real_y)) / self.batch_size
        L2 = -F.sum(F.log(1 - fake_y)) / self.batch_size
        loss = L1 + L2

        fake_prob = F.sum(fake_y) / self.batch_size

        chainer.report({'loss': loss, 'fake_prob':fake_prob}, self.dis)
        return loss

    def generator_loss(self, fake_y):
        loss = -F.sum(F.log(fake_y)) / self.batch_size

        chainer.report({'loss': loss}, self.gen)
        return loss

    def generate_real_img(self):
        batch = self.get_iterator('main').next()
        img_batch, real_label = self.converter(batch, self.device)

        x_real_img = Variable(img_batch)

        #Conditional Data
        x_real_label = Variable(label2images(real_label, self.class_num, self.img_shape))
        x_real_img_label = F.concat((x_real_img, x_real_label), axis=1)
        return x_real_img_label

    def generate_fake_img(self, is_random=False):
        noise_zy, fake_label = self.gen.generate_noise(self.device, self.batch_size, is_random)

        x_fake_img = self.gen(noise_zy)

        #Conditional Data
        x_fake_label = Variable(label2images(fake_label, self.class_num, self.img_shape))
        x_fake_img_label = F.concat((x_fake_img, x_fake_label), axis=1)
        return x_fake_img_label

    def update_core(self):
        for _ in range(self.dis_ln):
            x_real = self.generate_real_img()
            x_fake = self.generate_fake_img(is_random=self.is_random)
            y_real = self.dis(x_real)
            y_fake = self.dis(x_fake)
            dis_loss = self.discriminator_loss(y_real, y_fake)

            self.dis.cleargrads()
            dis_loss.backward()
            self.dis_opt.update()

        x_fake = self.generate_fake_img(is_random=self.is_random)
        y_fake = self.dis(x_fake)
        gen_loss = self.generator_loss(y_fake)

        self.gen.cleargrads()
        gen_loss.backward()
        self.gen_opt.update()