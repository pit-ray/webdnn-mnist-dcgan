#coding: utf-8
import chainer
from chainer import iterators, optimizers, datasets, training, optimizer_hooks
import chainer.training.extensions as ex

from networks import Generator, Discriminator
from updater import GANUpdater

def main():
    #hyper parameter
    z_dim = 100
    dis_ln = 1
    grad_clip = 0.1
    adam_alpha = 1e-4
    adam_beta1 = 0.5
    adam_beta2 = 0.9

    #training option
    is_random = False #whether conditional
    npz_interval = 100
    max_epoch = 200
    out_dir = 'result-dcgan'
    batch_size = 128
    device = 0

    gen_npz = None
    dis_npz = None
    #gen_npz = 'gen_snapshot_epoch-200.npz'
    #dis_npz = 'dis_snapshot_epoch-200.npz'

    train, _ = datasets.mnist.get_mnist(ndim=3)
    train_iter = iterators.SerialIterator(train, batch_size)

    gen = Generator(z_dim)
    gen.to_gpu(device=device)
    if gen_npz is not None:
        chainer.serializers.load_npz(out_dir + '/' + gen_npz, gen)
    gen_opt = optimizers.Adam(alpha=adam_alpha, beta1=adam_beta1, beta2=adam_beta2)
    gen_opt.setup(gen)
    gen_opt.add_hook(optimizer_hooks.GradientClipping(grad_clip))

    dis = Discriminator()
    dis.to_gpu(device=device)
    if dis_npz is not None:
        chainer.serializers.load_npz(out_dir + '/' + dis_npz, dis)
    dis_opt = optimizers.Adam(alpha=adam_alpha, beta1=adam_beta1, beta2=adam_beta2)
    dis_opt.setup(dis)
    dis_opt.add_hook(optimizer_hooks.GradientClipping(grad_clip))

    updater = GANUpdater(dis_ln=dis_ln, is_random=is_random, iterator=train_iter,
        optimizer={'gen': gen_opt, 'dis': dis_opt}, device=device)

    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out=out_dir)

    trainer.extend(ex.LogReport(log_name=None, trigger=(1, 'iteration')))
    trainer.extend(ex.PrintReport(['epoch', 'iteration', 'gen/loss', 'dis/loss', 'dis/fake_prob', 'elapsed_time']))
    trainer.extend(ex.PlotReport(['gen/loss', 'dis/loss'], x_key='epoch', file_name='loss.png',
        postprocess=lambda *args : gen.save_img(out_dir=out_dir + '/img')))
    trainer.extend(ex.PlotReport(['dis/fake_prob'], x_key='epoch', file_name='probability.png'))
    trainer.extend(ex.snapshot_object(gen, 'gen_snapshot_epoch-{.updater.epoch}.npz'),
        trigger=(npz_interval, 'epoch'))
    trainer.extend(ex.snapshot_object(dis, 'dis_snapshot_epoch-{.updater.epoch}.npz'),
        trigger=(npz_interval, 'epoch'))

    trainer.run()

if __name__ == '__main__':
    main()