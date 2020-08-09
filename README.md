# Wasserstein GAN (WGAN) with gradient penalty (WGAN-GP) Tensorflow 2 implementation

This repo is offering a perfect example for implementing WAGN-GP using Tensorflow 2.0. The code should be compatible with tf 2.2.0 or future versions.
The script is easy to follow for the beginners since I did not use complicated python structures. 

Original paper is https://arxiv.org/abs/1704.00028

**Data set**

The dataset is in pickle form, one can open it by

```
real_mask = pickle.load( open( "mask_data.pkl", "rb" ) )
```
This data is a set of masks, which would be used for language model or signal model with different lengths. Say, if we hope to generate the signals with variant length, we can train a gan for generating the mask first. The values of the mask are either '0' or '1' in float32 form. The maximum length is 8192 (2^13). Some examples are shown as follow


<p align="center">
  <img width="460" height="300" src="https://github.com/bigmao8576/WGAN_GP_MASK/blob/master/real_data.png">
</p>
