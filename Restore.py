from __future__ import print_function, division
import numpy as np
np.random.seed(1234)
# import time
import matplotlib.pyplot as plt
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(1234)
# from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.datasets import mnist
# from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Conv2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
def Score_compute(model):
    noise = np.random.normal(0, 1, (5000, 100))
    gen_imgs = model.predict(noise)
    # gen_imgs = 0.5 * gen_imgs + 0.5
    # print(gen_imgs[1])
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    idx = np.random.randint(0, X_train.shape[0], 5000)
    imgs = X_train[idx]
    Fake = gen_imgs.copy()
    Real = imgs.copy()
    # need to transfer to torch.array first.
    Mxx = distance(Real, Real)
    Mxy = distance(Real, Fake)
    Myy = distance(Fake, Fake)
    Score = mmd(Mxx, Mxy, Myy, 1)
    return Score
    #The below is fetched from GAN metric:
#1. Compute distance between X and Y , inputs are from convnet_feature_saver.
def distance(X, Y):
    nX = X.shape[0]
    nY = Y.shape[0]
    X.resize((nX,784))
    Y.resize((nY,784))
    X2 = np.sum((X* X),axis = 1)
    Y2 = np.sum((Y* Y),axis = 1)
    X2 = np.tile(X2,(nX,1))
    Y2 = np.tile(Y2,(nY,1))
    M = np.zeros((nX, nY))
    Q = np.copy(X2 + Y2.T)
    P = 2* np.dot(X, Y.T)
    M = np.subtract(Q, P)
    del nX, nY, X, Y, X2, Y2, Q, P
    return M
#2. Compute MMD, imput are Distance of real to real; Distance of real to fake, Distance of fake to fake. sigma =1.
def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean()
    Mxx = np.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = np.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = np.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = np.sqrt(np.absolute(Mxx.mean() + Myy.mean() - 2 * Mxy.mean()))
    return mmd

def sample_images(model, extract_path):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = model.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    cwd = os.getcwd()
    path = os.path.join(cwd, extract_path)
    if not os.path.isdir(path):
        os.makedirs(path)
    fig.savefig(path + "/fake_image.png")
    plt.close()
if __name__ == '__main__':
    seed = 1234
    tf.set_random_seed(seed)
    np.random.seed(seed)

    print("Start to restore the model")

    extract_path = 'FreezeGheadDtail70_4000'
    cwd = os.getcwd()
    path = os.path.join(cwd, extract_path)
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    model = load_model(path + "/G_model.h5")
    MMD = Score_compute(model)
    print("MMD Score is : %.4f" % MMD)
    sample_images(model, path)

    extract_path = 'NonFreeze_4000'
    cwd = os.getcwd()
    path = os.path.join(cwd, extract_path)
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    model = load_model(path + "/G_model.h5")
    MMD = Score_compute(model)
    print("MMD Score is : %.4f" % MMD)
    sample_images(model, path)

    extract_path = 'NonFreeze_8000'
    cwd = os.getcwd()
    path = os.path.join(cwd, extract_path)
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    model = load_model(path + "/G_model.h5")
    MMD = Score_compute(model)
    print("MMD Score is : %.4f" % MMD)
    sample_images(model, path)