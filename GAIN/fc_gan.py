import torch
from torch import nn,optim
from torch.autograd.variable import Variable
from torchvision import transforms,datasets
from utils import Logger
from IPython import display
import numpy as np
import matplotlib.pyplot as plt

from networks import *


def temper_data(flag="train"):
    if flag == "train":
        return torch.load('dataloader_temp_train2_new.pt')
        # return torch.load('../dataloader_temp_train2.pt')
    elif flag == 'test':
        return torch.load('dataloader_temp_test2_new.pt')
        # return torch.load('../dataloader_temp_test2.pt')
    else:
        raise NameError('No such data loader')

train_dataloader = temper_data()
num_batches_train = len(train_dataloader)
test_dataloader = temper_data('test')
num_batches_test = len(test_dataloader)


# 3
def images_to_vectors(images):
    return images.view(images.size(0), 133)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 19, 7)

def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda() 
    return n


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data




def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

if __name__ == "__main__":
    discriminator = DiscriminatorNet()
    generator = GeneratorNet()
    loss = nn.BCELoss()

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        loss.cuda()

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # Loss function

    # Number of steps to apply to the discriminator
    d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
    # Number of epochs
    num_epochs = 400

    num_test_samples = 16
    test_noise = noise(num_test_samples)

    logger = Logger(model_name='FCGAN', data_name='SensorScope')

    for epoch in range(num_epochs):
        for n_batch, (real_batch) in enumerate(train_dataloader):

            # 1. Train Discriminator
            real_data = Variable(images_to_vectors(real_batch))
            if torch.cuda.is_available(): real_data = real_data.cuda()
            # Generate fake data
            fake_data = generator(noise(real_data.size(0))).detach()
            # fake_data = generator(noise(real_data.size(0)))
            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                    real_data, fake_data)

            # 2. Train Generator
            # Generate fake data
            fake_data = generator(noise(real_batch.size(0)))
            # Train G
            g_error = train_generator(g_optimizer, fake_data)
            # Log error
            logger.log(d_error, g_error, epoch, n_batch, num_batches_train)

            # Display Progress
            if (n_batch) % 10 == 0:
                display.clear_output(True)
                # Display Images
                test_images = vectors_to_images(generator(test_noise)).data.cpu()
                # logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches_train);
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches_train,
                    d_error, g_error, d_pred_real, d_pred_fake
                )
            # Model Checkpoints
            logger.save_models(generator, discriminator, epoch)