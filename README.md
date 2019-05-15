# PartialFreezeGAN
A partial-freeze technique towards fast DCGAN training (in Python, using keras framework). 

### MMD Metric

A metric to evaluate GAN, implemented using Gaussian kernel.

Reference

Borji, Ali. "Pros and cons of gan evaluation measures." Computer Vision and Image Understanding 179 (2019): 41-65.

### FreezeGAN

A technique to accelerate fast GAN convergence. Learnable parameters of tail of the discriminator and head of the generator which take the most part of overall trainable parameter is freezed, which is called partial-freeze. 

![Alt text](https://github.com/zlijingtao/PartialFreezeGAN/blob/master/Idea/process.PNG?raw=true "Fig. 1. Partial Freeze of the full potential model.")

After a certain fraction of time (1 - $\alpha$) training on the full-potential model, we then apply the partial-freeze for the rest of the time, to continuely train on a less potential model. This technique shows 100% less training time to achieve the same level of convergence based on MMD metric. And it also shows the potential to avoid overfitting which needs further investigation.

![Alt text](https://github.com/zlijingtao/PartialFreezeGAN/blob/master/Idea/process1.PNG?raw=true "Fig. 2. Training process of partial freeze GAN.")

The comparison of our proposed technique with the baseline (4,000 training steps with learning rate 0.0002, freeze ratio is set to 0.7 in FreezeGAN) shows a much better convergence.

![Alt text](https://github.com/zlijingtao/PartialFreezeGAN/blob/master/Idea/comparison.PNG?raw=true "Fig. 3. Comparison with a 4,000 steps baseline DCGAN.")

Without the partial-freeze technique, we need train excessive 8,000 steps to achieve same level of convergence.

![Alt text](https://github.com/zlijingtao/PartialFreezeGAN/blob/master/Idea/8000.PNG?raw=true "Fig. 4. 8,000 steps of baseline DCGAN to get the same convergence.")
