from __future__ import print_function, division
import numpy as np
np.random.seed(1234)
import time
import matplotlib.pyplot as plt
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(1234)
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

class GAN():
    def __init__(self, freeze_Ghead = False, freeze_Gtail = False, freeze_Dhead = False, freeze_Dtail = False, freeze_ratio = 0.9):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.forward_to_backward = freeze_ratio
        '''Decide whether to freeze the layer'''
        self.activate_Ghead = (freeze_Ghead == False)
        self.activate_Gtail = (freeze_Gtail == False)
        self.activate_Dhead = (freeze_Dhead == False)
        self.activate_Dtail = (freeze_Dtail == False)
        optimizer = Adam (0.0002, 0.5)
        '''The first parameter for Adam optimizer is learning rate, the second is for the momentum GD beta. beta = 0.5 means a low friction'''
        
        '''Build and compile the discriminator'''
        self.discriminator_head = self.build_discriminator_head()
        self.discriminator_tail = self.build_discriminator_tail()
        self.generator_head = self.build_generator_head()
        self.generator_tail = self.build_generator_tail() 
        
        '''Build and compile the discriminator_fr'''
        
        self.discriminator_head.trainable = self.activate_Dhead
        self.discriminator_tail.trainable = self.activate_Dtail
        ''''''
        r = Input(shape=self.img_shape)
        
        validity = self.discriminator_tail(self.discriminator_head(r))
        
        self.discriminator_fr = Model(r, validity)
        self.discriminator_fr.compile(loss='binary_crossentropy',
                                  optimizer=optimizer,
                                  metrics=['accuracy'])
        print("summary for freezed discriminator:")
        self.discriminator_fr.summary()
        
        '''Freeze the tail layer => False, Activate => True'''
        self.generator_tail.trainable = self.activate_Gtail
        self.generator_head.trainable = self.activate_Ghead
        '''Freeze the head layer => False, Activate => True'''
        z = Input(shape=(self.latent_dim,))
        
        img = self.generator_tail(self.generator_head(z))
        
        self.generator_fr = Model(z, img)
        self.generator_fr.compile(loss='binary_crossentropy', optimizer=optimizer)
        print("summary for freezed generator model:")
        self.generator_fr.summary()
        
        self.discriminator_fr.trainable = False
        
        validity = self.discriminator_fr(img)
        
        self.combined_fr = Model(z, validity)
        self.combined_fr.compile(loss='binary_crossentropy', optimizer=optimizer)
        print("summary for freezed combined model:")
        self.combined_fr.summary()        

        '''We will define the build.discriminator later'''
        self.discriminator_tail.trainable = True
        self.discriminator_head.trainable = True
        r = Input(shape=self.img_shape)
        
        validity = self.discriminator_tail(self.discriminator_head(r))

        self.discriminator = Model(r, validity)
        self.discriminator.compile(loss='binary_crossentropy',
                                  optimizer=optimizer,
                                  metrics=['accuracy'])
        print("summary for discriminator:")
        self.discriminator.summary()

        self.generator_tail.trainable = True
        self.generator_head.trainable = True
        z = Input(shape=(self.latent_dim,))
        
        img = self.generator_tail(self.generator_head(z))

        '''Build a trainable generator names generator'''

        self.generator = Model(z, img)
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)       
        print("summary for generator model:")
        self.generator.summary()
        
        self.discriminator.trainable = False
        
        validity = self.discriminator(img)
        '''Build a combined model for training the generator'''
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        print("summary for combined model:")
        self.combined.summary()
        
    def build_generator_head(self):

        noise = Input(shape=(self.latent_dim,))

        glayer1 = Dense(128 * 7 * 7, activation='relu', input_dim=self.latent_dim)(noise)
        greshape1 = Reshape((7, 7, 128))(glayer1)
        gup1 = UpSampling2D()(greshape1)

        return Model(noise, gup1)

    def build_generator_tail(self):    

        bodyg = Input(shape=(14, 14, 128))
                
        glayer2 = Conv2D(128, (3, 3), padding='same')(bodyg)
        gnormalization1 = BatchNormalization(momentum=0.8)(glayer2)
        gactivation1 = Activation("relu")(gnormalization1)

        gup2 = UpSampling2D()(gactivation1)
        glayer3 = Conv2D(64, (3, 3), padding='same')(gup2)
        gnormalization2 = BatchNormalization(momentum=0.8)(glayer3)
        gactivation2 = Activation("relu")(gnormalization2)
        glayer4 = Conv2D(1, (3, 3), padding='same')(gactivation2)
        gactivation3 = Activation("tanh")(glayer4)
        return Model(bodyg, gactivation3)
    
    def build_discriminator_head(self):

        img = Input(shape=self.img_shape)

        dlayer1 = Conv2D(32, (3, 3), strides=2, padding='same')(img)
        dactivation1 = LeakyReLU(alpha=0.2)(dlayer1)
        dropout1 = Dropout(0.25)(dactivation1)
        dlayer2 = Conv2D(64, (3, 3), strides=2, padding='same')(dropout1)
        dzeropad1 = ZeroPadding2D(padding=((0, 1), (0, 1)))(dlayer2)
        dnormalization1 = BatchNormalization(momentum=0.8)(dzeropad1)
        dactivation2 = LeakyReLU(alpha=0.2)(dnormalization1)
        dropout2 = Dropout(0.25)(dactivation2)
        dlayer3 = Conv2D(128, (3, 3), strides=2, padding='same')(dropout2)
        dnormalization2 = BatchNormalization(momentum=0.8)(dlayer3)
        dactivation3 = LeakyReLU(alpha=0.2)(dnormalization2)
        
        return Model(img, dactivation3)

    def build_discriminator_tail(self):

        bodyd = Input(shape=(4, 4, 128))

        dropout3 = Dropout(0.25)(bodyd)
        dlayer4 = Conv2D(256, (3, 3), strides=1, padding='same')(dropout3)
        dnormalization3 = BatchNormalization(momentum=0.8)(dlayer4)
        dactivation4 = LeakyReLU(alpha=0.2)(dnormalization3)
        dropout4 = Dropout(0.25)(dactivation4)
        flatten1 = Flatten()(dropout4)
        validity = Dense(1, activation='sigmoid')(flatten1)

        return Model(bodyd, validity)   
    
    def train(self, epochs, batch_size=128, sample_interval=50, extract_path = 'model'):

        #Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        MMD_list =[]
        count = 0
        for epoch in range(int(epochs*(1-self.forward_to_backward)+1)):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, extract_path)
                MMD = self.Score_compute()
                print("MMD Score is : %.4f" % MMD)
                MMD_list.append(MMD)
                count += 1
        if self.forward_to_backward != 0:
            print("Start to train on freezed model:")
        for epoch in range(int(epochs*(1-self.forward_to_backward)+1),epochs+1):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator_fr.predict(noise)

            # Train the discriminator
            # Half from training set, half from fake_sample
            d_loss_real = self.discriminator_fr.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator_fr.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined_fr.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images_fr(epoch, extract_path)
                MMD = self.Score_compute()
                print("MMD Score is : %.4f" % MMD)
                MMD_list.append(MMD)
                count += 1
        print("Start to plot MMD curve:")

        cwd = os.getcwd()
        path = os.path.join(cwd, extract_path)
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path + '/model_log.csv', 'a') as myfile:
            myfile.writelines("Final MMD: " + "\n" + str(MMD) + "\n")

        fig = plt.figure()
        plt.plot(MMD_list)
        plt.ylabel("MMD")
        plt.ylim(0, 0.1)
        plt.xlabel("Per " + str(sample_interval) + " Steps")
        fig.savefig(path + "/MMD.png")


    def sample_images(self, epoch, extract_path):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

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
        fig.savefig(path + "/%d.png" % epoch)
        plt.close()

    def sample_images_fr(self, epoch, extract_path):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator_fr.predict(noise)

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
        fig.savefig(path + "/%d.png" % epoch)
        plt.close()

    def Score_compute(self):
        noise = np.random.normal(0, 1, (5000, 100))
        gen_imgs = self.generator_fr.predict(noise)
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



if __name__ == '__main__':
    seed = 1234
    tf.set_random_seed(seed)
    np.random.seed(seed)


    # print("Start to train the FreezeDCGAN")
    #
    # extract_path = 'FreezeGheadDtail70_4000'
    # cwd = os.getcwd()
    # path = os.path.join(cwd, extract_path)
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    #
    # gan = GAN(freeze_Ghead=True, freeze_Gtail=False, freeze_Dhead=False, freeze_Dtail=True, freeze_ratio=0.7)
    # start_time = time.time()
    # gan.train(epochs=4000, batch_size=32, sample_interval=50, extract_path=extract_path)
    # end_time = time.time()
    # duration1 = end_time - start_time
    # print("Spend time: " + str(duration1))
    #
    # with open(path + '/model_log.csv', 'a') as myfile:
    #     myfile.writelines("Time Consuming: " + "\n" + str(duration1) + "\n")
    # gan.generator.save(path + "/G_model.h5")
    #
    # print("Start to train the conventional DCGAN")
    #
    # extract_path = 'NonFreeze_4000'
    # cwd = os.getcwd()
    # path = os.path.join(cwd, extract_path)
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    # gan1 = GAN(freeze_Ghead=False, freeze_Gtail=False, freeze_Dhead=False, freeze_Dtail=False, freeze_ratio=0.0)
    # start_time = time.time()
    # gan1.train(epochs=4000, batch_size=32, sample_interval=50, extract_path=extract_path)
    # end_time = time.time()
    # duration2 = end_time - start_time
    # print("Spend time: " + str(duration2))
    #
    # with open(path + '/model_log.csv', 'a') as myfile:
    #     myfile.writelines("Time Consuming: " + "\n" + str(duration2) + "\n")
    # gan1.generator.save(path + "/G_model.h5")


    print("Start to train the conventional DCGAN")

    extract_path = 'NonFreeze_8000'
    cwd = os.getcwd()
    path = os.path.join(cwd, extract_path)
    if not os.path.isdir(path):
        os.makedirs(path)
    gan2 = GAN(freeze_Ghead=False, freeze_Gtail=False, freeze_Dhead=False, freeze_Dtail=False, freeze_ratio=0.0)
    start_time = time.time()
    gan2.train(epochs=8000, batch_size=32, sample_interval=50, extract_path=extract_path)
    end_time = time.time()
    duration3 = end_time - start_time
    print("Spend time: " + str(duration3))

    with open(path + '/model_log.csv', 'a') as myfile:
        myfile.writelines("Time Consuming: " + "\n" + str(duration3) + "\n")
    gan2.generator.save(path + "/G_model.h5")
    print("Training End!")