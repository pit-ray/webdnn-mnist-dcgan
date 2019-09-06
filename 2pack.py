#coding: utf-8
import numpy as np
import chainer

from webdnn.frontend.chainer import ChainerConverter
from webdnn.backend import generate_descriptor

from networks import Generator

def main():
    z_dim = 100
    device = -1 #CPU
    batch_size = 1
    model = Generator(z_dim)

    model.to_gpu()
    chainer.serializers.load_npz('result-dcgan/gen_snapshot_epoch-200.npz', model)
    model.to_cpu()

    x, _ = model.generate_noise(device, batch_size)
    y = model(x)

    graph = ChainerConverter().convert([x], [y])
    exec_info = generate_descriptor("webassembly", graph)
    exec_info.save("./model")

if __name__ == '__main__':
    main()