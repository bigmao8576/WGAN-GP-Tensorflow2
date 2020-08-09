#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Shitong Mao
University of Pittsburgh 
bigmao_8576@hotmail.com

WGAN-GP

This code is not properly orgnized, just used for proof-of-concept purpose
I have made my best to make it clear and easy to read

"""

import numpy as np
import tensorflow as tf
import pickle


import matplotlib.pyplot as plt  


from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

# read dataset
real_mask = pickle.load( open( "mask_data.pkl", "rb" ) )


########================================================================================
# set up some parameters
data_params = {'data_len':8192,'nb_channel':1,'data_size':1698}
network_params = {'latent_dim':1000}


batch_size = 512
# just for grad clipping
# I find wgan-gp still has the issue of grad exploding 
grad_norm_th = 50.0
########================================================================================



class mask_gan(Model):
  def __init__(self,data_params):
    super(mask_gan, self).__init__()


    self.d1 = Dense(4096)
    self.d2 = Dense(data_params['data_len'],activation='sigmoid')

  def call(self, x):

    #[batch_size, latent_dim] =>[batch_size, 4096]
    x = self.d1(x)
    x = tf.nn.leaky_relu(x)
    
    #[batch_size, 4096] =>[batch_size, 8192]    
    x = self.d2(x)
    #[batch_size,8192] =>[batch_size,8192,1]
    x = tf.expand_dims(x,-1)
    return x


class mask_dis(Model):
  def __init__(self):
    super(mask_dis, self).__init__()

    self.d1 = Dense(4096)
    self.d2 = Dense(1024)
    self.d3 = Dense(1)

  def call(self, x):
    #[batch_size, 8192,1] =>[batch_size, 8192]
    x = tf.squeeze(x,-1)
    
    #[batch_size, 8192] =>[batch_size, 4096]
    x = self.d1(x)
    x = tf.nn.leaky_relu(x)
    
    #[batch_size, 256] =>[batch_size, 1024]
    x = self.d2(x)
    x = tf.nn.leaky_relu(x)
    
    #[batch_size, 256] =>[batch_size, 1]
    x = self.d3(x)
    return x



# Create an instance of the model
mask_generator = mask_gan(data_params)
mask_discriminator = mask_dis()


gen_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.RMSprop(5e-5)


def gradient_penalty(x, x_fake):
    '''
    1. get epsilon
    2. calculate the x_hat
    3. get gradient
    4. regularizer
    
    input:
        x: real data
        x_fake: fake data
        
        x and x_fake must have the same batch size
    
    '''
    
    if x.shape[0]!=x_fake.shape[0]:
        raise ValueError('x and x_fake must have the same batch size')
    
    
    # be careful, the epsilon is applied on each samples.
    # if thhe shpe of the sample is [batch_size, a,b,c,d]
    # the epsilon should have the shape of [batch_size,1,1,1,1]
    temp_shape = [x.shape[0]]+[1 for _ in  range(len(x.shape)-1)]
    epsilon = tf.random.uniform(temp_shape, 0.0, 1.0)
    x_hat = epsilon * x + (1 - epsilon) * x_fake
    
    # gradient
    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat = mask_discriminator(x_hat)
    gradients = t.gradient(d_hat, x_hat)
    
    # be carefule, the L2 norm is calculated on each sample
    # and then averaged through all the samples
    g_norm2 = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
    d_regularizer = tf.reduce_mean((g_norm2 - 1.0) ** 2)
    return d_regularizer

def compute_g_loss(x):
    
    '''
    the input x is the real data, here only used for the shape.
    This is a silly way, but I am a little bit lazy...
    '''

    z_noise = tf.random.uniform([x.shape[0], network_params['latent_dim']])
    x_fake = mask_generator(z_noise)
    logits_x_fake = mask_discriminator(x_fake)

    # Be careful, when training g, we need the probability of x_fake 
    # as large as possible. So when the loss is reducing, we need "-" in front of 
    # x_fake to implement that
    g_loss = -tf.reduce_mean(logits_x_fake)

    return g_loss

def compute_d_loss(x):


    z_noise = tf.random.uniform([x.shape[0], network_params['latent_dim']])
    x_fake = mask_generator(z_noise)

    logits_x = mask_discriminator(x)
    logits_x_fake = mask_discriminator(x_fake)

    d_regularizer = gradient_penalty(x, x_fake)
    
    
    # loss calculation. same with the original paper
    # lambda is 10
    d_loss = tf.reduce_mean(logits_x_fake) - tf.reduce_mean(logits_x) + 10*d_regularizer 
    
    return d_loss

@tf.function
def train_g_step(real_data,network_params):
    
    # only use the shape of real date
    noise = tf.random.uniform([real_data.shape[0], network_params['latent_dim']])

    with tf.GradientTape() as gen_tape:

        gen_loss = compute_g_loss(noise)

    gradients_of_generator = gen_tape.gradient(gen_loss, mask_generator.trainable_variables)
    
    # I found "clipnorm" in tf.keras.optimizers.Adam() does work correctly
    # so I used this one to clip the gradient.
    # Also, tf.clip_by_global_norm() may also give you a gradient norm, 
    # but I found it is different from tf.linalg.global_norm()
    # I trust tf.linalg.global_norm()
    gradients_of_generator,_ = tf.clip_by_global_norm(gradients_of_generator,grad_norm_th)
    
    gen_optimizer.apply_gradients(zip(gradients_of_generator, mask_generator.trainable_variables))

    gradient_norm = tf.linalg.global_norm(gradients_of_generator)
    
    return gradient_norm

@tf.function
def train_d_step(real_data,network_params):

    with tf.GradientTape() as disc_tape:
        disc_loss = compute_d_loss(real_data)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, mask_discriminator.trainable_variables)
    
    # I found "clipnorm" in tf.keras.optimizers.Adam() does work correctly
    # so I used this one to clip the gradient.
    # Also, tf.clip_by_global_norm() may also give you a gradient norm, 
    # but I found it is different from tf.linalg.global_norm()
    # I trust tf.linalg.global_norm()
    gradients_of_discriminator,_ = tf.clip_by_global_norm(gradients_of_discriminator,grad_norm_th)
    
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, mask_discriminator.trainable_variables))
    
    gradient_norm = tf.linalg.global_norm(gradients_of_discriminator)
    
    return gradient_norm

def evaluation(x):
    
    '''
    just used for evaluation
    one can also ensemble this function with train_d/g_step
    '''
    
    desi_real = mask_discriminator(x)
    d_r_b_loss = np.mean(desi_real)
    
    z_noise = tf.random.uniform([x.shape[0], network_params['latent_dim']])
    x_fake = mask_generator(z_noise)
    desi_fake = mask_discriminator(x_fake)
    d_f_b_loss = np.mean(desi_fake)
    
    return d_r_b_loss,d_f_b_loss

def get_flag(examples):
    '''
    give a metric to stop training.
    
    the differentiate of the desired fake mask should be 
    
    1. maximum = 0
    2. minimum = -1
    3. each mask has only one -1 differentiate
    
    (I am thinking to add more)
    '''
    
    
    a = np.squeeze(np.int64(np.round(examples)))
    c = np.diff(a)
    b = (-np.sum(c))/examples.shape[0]
    
    if np.min(c) == -1.0 and np.max(c) == 0.0 and b == 1.0:
        return True
    else:
        return False



# data pipeline
r_mask = tf.data.Dataset.from_tensor_slices((real_mask))
r_mask = r_mask.shuffle(10000)
r_mask = r_mask.batch(batch_size)

# if one hopes to fetch one batch just for debugging, use the following code
# =============================================================================
# temp_iter = r_mask.__iter__()
# x = temp_iter.__next__()
# 
# 
# z_noise = tf.random.uniform([x.shape[0], network_params['latent_dim']])
# x_fake = mask_generator(z_noise)
# desi_fake = mask_discriminator(x_fake)
# desi_fake = desi_fake.numpy() 
# 
# x_fake = x_fake.numpy()
# desi_real = mask_discriminator(x)
# desi_real = desi_real.numpy() 
# 
# =============================================================================


examp_noise = tf.random.uniform([batch_size, network_params['latent_dim']])
total_d_r_loss = []
total_d_f_loss = []
flag = False
epoch = 0 


while not flag:
    
    ep_d_r_loss = []
    ep_d_f_loss = []
    
    for data_batch in r_mask:
        
        #train d
        temp_d_norm = train_d_step(data_batch,network_params)
        d_r_b_loss1,d_f_b_loss1 = evaluation(data_batch) 
        
        #train g
        temp_g_norm = train_g_step(data_batch,network_params)    
        d_r_b_loss2,d_f_b_loss2 = evaluation(data_batch) 
        

        
        d_r_b_loss = (d_r_b_loss1+d_r_b_loss2)/2
        d_f_b_loss = (d_f_b_loss1+d_f_b_loss2)/2
        
        ep_d_r_loss.append(d_r_b_loss)
        ep_d_f_loss.append(d_f_b_loss)
        
    ep_d_r_loss = np.mean(ep_d_r_loss)
    ep_d_f_loss = np.mean(ep_d_f_loss)
     
    if epoch%10==0:
        
        
        
        total_d_r_loss.append(ep_d_r_loss)
        total_d_f_loss.append(ep_d_f_loss)
                
    # try some examples
        
         
        examples = mask_generator(examp_noise)
        examples = np.squeeze(examples.numpy())
        flag = get_flag(examples)    
        print('EP %d, d_r_loss = %0.4f, d_f_loss = %0.4f, flag = %d'%(epoch, ep_d_r_loss, ep_d_f_loss, np.int64(flag)))
        
        plt.plot(examples[:6,:].T)
        plt.show()
        
        fig2 = plt.figure(2)

        plt.plot(total_d_r_loss,label='d_r_loss')
        plt.plot(total_d_f_loss,label='d_f_loss')
        plt.legend()
        plt.show()
        

    
    epoch += 1   
        
mask_generator.save('saved_model/mask_gan1')