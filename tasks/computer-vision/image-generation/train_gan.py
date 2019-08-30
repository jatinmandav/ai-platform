#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.keras

from DataLoader import DataLoader
from model import DCGAN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_rows', '-r', help='No of Rows in Image, Default=28', default=28, type=int)
parser.add_argument('--img_cols', '-c', help='No of Cols in Image, Default=28', default=28, type=int)
parser.add_argument('--channels', '-ch', help='No of Channels in Image, Default=1', default=1, type=int)
parser.add_argument('--noise_dim', '-n', help='Dimension of Noise, Default=100', default=100, type=int)
parser.add_argument('--epoch', '-e', help='No of Epochs to train on, Default=3000', default=3000, type=int)
parser.add_argument('--batch', '-b', help='Batch Size, Default=128', default=128, type=int)

args = parser.parse_args()

mlflow.keras.autolog()

params = {'img_rows': args.img_rows, 'img_cols': args.img_cols, 'channels': args.channels,
          'noise_dim': args.noise, 'epoch': args.epoch, 'plt_frq': 20, 'batch_size': args.batch}


data_loader = DataLoader(noise_dim=params['noise_dim'])
data_loader.load_real()

data_loader.dataset.shape


indexes = np.random.randint(0, data_loader.dataset.shape[0], 16)
images = data_loader.dataset[indexes]

dim = (4, 4)

dcgan_model = DCGAN(img_rows=params['img_rows'], img_cols=params['img_cols'], channels=params['channels'],
                    noise_dim=params['noise_dim'])
dcgan_model.build_gan()


print('\nGENERATOR: ')
dcgan_model.generator.summary()

print('\nDISCRIMINATOR: ')
dcgan_model.discriminator.summary()


def plot_loss(losses):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()

def plot_gen(epoch=1, n_ex=16,dim=(4,4), figsize=(7,7)):
    noise = np.random.uniform(0,1,size=[n_ex,100])
    generated_images = dcgan_model.generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = np.reshape(generated_images[i], (28, 28))
        plt.imshow(img, cmap='Greys')
        plt.axis('off')
    plt.tight_layout()
    if epoch != None:
        plt.savefig('generated_samples/{}.png'.format(epoch))
    plt.show()

losses = {'g': [], 'd': []}

config = {'img_rows': args.img_rows, 'img_cols': args.img_cols, 'channels': args.channels,
          'noise_dim': args.noise, 'epoch': args.epoch, 'batch_size': args.batch}


client = mlflow.tracking.MlflowClient()
with mlflow.start_run() as run:
    for key in config.keys():
        mlflow.log_param(key, config[key])

    client.set_tag(run.info.run_id, "experiment_name", "FashionMNIST_generation")
    for epoch in range(params['epoch']):
        images, real_labels = data_loader.next_real_batch(params['batch_size'])
        generated_images, fake_labels = data_loader.next_fake_batch(dcgan_model.generator, params['batch_size'])

        d_loss_real = dcgan_model.discriminator.train_on_batch(images, real_labels)
        d_loss_fake = dcgan_model.discriminator.train_on_batch(generated_images, fake_labels)

        d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
        losses['d'].append(d_loss[0])

        # Generator
        noise = data_loader.generate_noise(params['batch_size'])
        g_loss = dcgan_model.dcgan.train_on_batch(noise, real_labels)
        losses['g'].append(g_loss)

        mlflow.log_metric("generator_loss", g_loss)
        mlflow.log_metric("discriminator_loss", d_loss[0])

        if epoch%params['plt_frq'] == params['plt_frq'] - 1:
            print('Generator Loss: {}, Discriminator Loss: {}'.format(g_loss, d_loss[0]))
            #plot_loss(losses)
            #plot_gen(epoch=epoch)

plot_gen(epoch=None)

training_info = client.get_run(run.info.run_id)
mlflow.keras.log_model(dcgan_model.generator, "model", custom_objects=config)

dcgan_model.generator.save_weights('generator.weights')
dcgan_model.discriminator.save_weights('discriminator.weights')
dcgan_model.dcgan.save_weights('dcgan.weights')
