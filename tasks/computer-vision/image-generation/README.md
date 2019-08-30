# Image Generation

### Deep Convolutional Generative Adversarial Networks (DCGAN)

DCGAN used to generate Fashion-MNIST images from Noise.

Link to Paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

Dataset: [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)


### Code

 - `DataLoader.py` reads the dataset and performs preprocessing on images and generates batches for training/testing.

 - DCGAN model is defined in `model.py`. Modify the architecture of generator/discriminator for getting different results.

 - `Mlflow` logs the parameters and loss metrics during each training phase.

 - Fire-up `jupyter-notebook` and run-all [notebook](Deep_Convolutional_Generative_Adversarial_Networks_Fashion_MNIST.ipynb) or run `train_gan.py` to train via command-line.


For more details and to change parameters, run following command:
 ```
 python3 train_gan.py --help
 usage: train_gan.py [-h] [--img_rows IMG_ROWS] [--img_cols IMG_COLS]
                     [--channels CHANNELS] [--noise_dim NOISE_DIM]
                     [--epoch EPOCH] [--batch BATCH]

 optional arguments:
   -h, --help            show this help message and exit
   --img_rows IMG_ROWS, -r IMG_ROWS
                         No of Rows in Image, Default=28
   --img_cols IMG_COLS, -c IMG_COLS
                         No of Cols in Image, Default=28
   --channels CHANNELS, -ch CHANNELS
                         No of Channels in Image, Default=1
   --noise_dim NOISE_DIM, -n NOISE_DIM
                         Dimension of Noise, Default=100
   --epoch EPOCH, -e EPOCH
                         No of Epochs to train on, Default=3000
   --batch BATCH, -b BATCH
                         Batch Size, Default=128
 ```
