# Image Generation

### Deep Convolutional Generative Adversarial Networks (DCGAN)

DCGAN used to generate Fashion-MNIST images from Noise.

Link to Paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

Dataset: [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)


### Code

 - `DataLoader.py` reads the dataset and performs preprocessing on images and generates batches for training/testing.

 - DCGAN model is defined in `model.py`. Modify the architecture of generator/discriminator for getting different results.

 - `Mlflow` logs the parameters and loss metrics during each training phase.
