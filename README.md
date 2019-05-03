# PartialFreezeGAN
A partial-freeze technique towards fast GAN training

### MMD Metric

A metric to evaluate GAN.

Reference

Borji, Ali. "Pros and cons of gan evaluation measures." Computer Vision and Image Understanding 179 (2019): 41-65.

### FreezeGAN

A technique to accelerate fast GAN convergence. Learnable parameters of tail of the discriminator and head of the generator which take the most part of overall trainable parameter is freezed, which is called partial-freeze. 

After a certain fraction of time training on the full-potential model, we then apply the partial-freeze to the model, to continuely train on a less potential model. This technique shows 100% less training time to achieve the same level of convergence based on MMD metric. And it also shows the potential to avoid overfitting which needs further investigation.

The comparison of our proposed technique with the baseline is:
