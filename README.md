# Wasserstein GAN (WGAN) with gradient penalty (WGAN-GP) Tensorflow 2 implementation

This repo is offering a perfect example for implementing WAGN-GP using Tensorflow 2.0. The code should be compatible with tf 2.2.0 or future versions.
The script is easy to follow for the beginners since I did not use complicated python structures. More dependencies can be found in Pipfile.

Original paper is https://arxiv.org/abs/1704.00028

**Data set**

The dataset is in pickle form, one can open it by

```
real_mask = pickle.load( open( "mask_data.pkl", "rb" ) )
```
This data is a set of masks, which would be used for language model or signal model with different lengths. Say, if we hope to generate the signals with variant length, we can train a gan for generating the mask first. The values of the mask are either '0' or '1' in float32 form. The maximum length is 8192 (2^13). Some examples are shown as follows


<p align="center">
  <img width="460" height="300" src="https://github.com/bigmao8576/WGAN_GP_MASK/blob/master/real_data.png">
</p>

**Model**

1. Generator: 

The dimension of the latent vector is 1000. Considering the mask is a 1d vector, we can just use two dense layers. The activation function of the first one is leaky_relu, and the second one is sigmoid, since the output is either 1 or 0. 

2. Critic:

Just three dense layers. The activation functions of the first two layers are leaky_relu. **Some online documents used sigmoid as the last layer activation for critic, but we found unstability during trainning ('nan' values).** So no activation function is used for the last layer.

**Training**

1. I found **gradient exploding** when training WGAN. So I implemented gradient-clipping.
2. For generater, the optimizer is Adam, while RMSprop for critic.
3. A flag function was used for iteratively examining the generated data, once these data meet some criterion, the training stops.

The training curves are shown as follows:
<p align="center">
  <img width="460" height="300" src="https://github.com/bigmao8576/WGAN_GP_MASK/blob/master/training%20curves.png">
</p>

4. Learning rate decay may also be needed in the future.

**Results**

1. The generated masks are shown as follows:

<p align="center">
  <img width="460" height="300" src="https://github.com/bigmao8576/WGAN_GP_MASK/blob/master/examples.png">
</p>


They still have some 'noise', but a 'np.round()' function is enough for future usage.

2. Diversity 

The easist way to examine diversity is to plot the histogram of the lengths of masks (summation of '1's). The length distributions of real data(1600+) and fake data(512) are shown as follows:


<p align="center">
  <img width="460" height="300" src="https://github.com/bigmao8576/WGAN_GP_MASK/blob/master/real_data_length_distribution.png">
</p>

<p align="center">
  <img width="460" height="300" src="https://github.com/bigmao8576/WGAN_GP_MASK/blob/master/fake_data_length_distribution.png">
</p>


