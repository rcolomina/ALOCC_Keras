from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import InputLayer


from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import losses
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import keras.backend as K
import scipy
import logging
import matplotlib.pyplot as plt
import os
import cv2

import numpy as np

from utils import *
from kh_tools import *


def load_input_data(input_folder, randomisation):            
    from landsat_data_loader import LandsatDataLoader
    # EXPERIMENT 3
    #input_folder = root + '/TDS_NOVELTY_DETECTION/EXP_03/nominal_chips/'
    # EXPERIMENT 4
    print("Loading Data from {} Randomisation {}".format(input_folder,randomisation))
    landsatDataLoader = LandsatDataLoader(input_folder)            
    X_train = landsatDataLoader.load_data(randomisation)
    X_train = X_train / 255
    print("Data Loaded and Normalised")
    return X_train
    

class ALOCC_Model():
    def __init__(self,
               input_height=28,input_width=28,
               output_height=28, output_width=28,
               attention_label=1, is_training=True,
               z_dim=100, gf_dim=16, df_dim=16, c_dim=3,
               dataset_name=None, dataset_address=None, input_fname_pattern=None,
               checkpoint_dir='checkpoint', log_dir='log', sample_dir='sample', r_alpha = 0.2,
               kb_work_on_patch=True, nd_patch_size=(10, 10), n_stride=1,
               n_fetch_data=10, randomization=False, train_data=None):
        """
        This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection.
        :param sess: TensorFlow session.
        :param input_height: The height of image to use.
        :param input_width: The width of image to use.
        :param output_height: The height of the output images to produce.
        :param output_width: The width of the output images to produce.
        :param attention_label: Conditioned label that growth attention of training label [1]
        :param is_training: True if in training mode.
        :param z_dim:  (optional) Dimension of dim for Z, the output of encoder. [100]
        :param gf_dim: (optional) Dimension of gen filters in first conv layer, i.e. g_decoder_h0. [16] 
        :param df_dim: (optional) Dimension of discrim filters in first conv layer, i.e. d_h0_conv. [16] 
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        :param dataset_name: 'UCSD', 'mnist' or custom defined name.
        :param dataset_address: path to dataset folder. e.g. './dataset/mnist'.
        :param input_fname_pattern: Glob pattern of filename of input images e.g. '*'.
        :param checkpoint_dir: path to saved checkpoint(s) directory.
        :param log_dir: log directory for training, can be later viewed in TensorBoard.
        :param sample_dir: Directory address which save some samples [.]
        :param r_alpha: Refinement parameter, trade-off hyperparameter for the G network loss to reconstruct input images. [0.2]
        :param kb_work_on_patch: Boolean value for working on PatchBased System or not, only applies to UCSD dataset [True]
        :param nd_patch_size:  Input patch size, only applies to UCSD dataset.
        :param n_stride: PatchBased data preprocessing stride, only applies to UCSD dataset.
        :param n_fetch_data: Fetch size of Data, only applies to UCSD dataset. 
        """

        self.b_work_on_patch = kb_work_on_patch
        self.sample_dir = sample_dir

        self.is_training = is_training

        self.r_alpha = r_alpha

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.dataset_name = dataset_name
        self.dataset_address= dataset_address
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.attention_label = attention_label
        if self.is_training:
            logging.basicConfig(filename='ALOCC_loss.log', level=logging.INFO)

        #self.data
        #def load_input_data(self, input_folder, randomisation):            
        #if self.dataset_name == 'mnist':            
        #from landsat_data_loader import LandsatDataLoader
        #root = "/QCOLT/QCOLT_DEV_OPS/"
        #root = "/media/rcolomina/Toshiba640"
        # EXPERIMENT 3
        #input_folder = root + '/TDS_NOVELTY_DETECTION/EXP_03/nominal_chips/'
        # EXPERIMENT 4
        #input_folder = root + '/TDS_NOVELTY_DETECTION/EXP_04/nominal_chips/'
        #print("Loading Data")
        #landsatDataLoader = LandsatDataLoader(input_folder)            
        #X_train = landsatDataLoader.load_data(randomisation)
        #X_train = X_train / 255
        #print("Data Has been Loaded")

        self.setup_train_data = False
        
        self.c_dim = 1
            
        #(X_train, y_train), (_, _) = mnist.load_data()              
        #Make the data range between 0~1.
        #X_train = X_train / 255
        #specific_idx = np.where(y_train == self.attention_label)[0]
        #self.data = X_train[specific_idx].reshape(-1, 28, 28, 1)
        #self.c_dim = 1
            
        #elif self.dataset_name == 'landsat':
        #    pass
        #else:
        #    assert('Error in loading dataset')
        
        self.grayscale = (self.c_dim == 1)
        self.data_loaded = True
        self.build_model()
        self.model_built = True

    def set_train_data(self,train_data):        
        X_train = train_data
        self.data = X_train.reshape(-1,28,28,1)
        self.setup_train_data = True
            
        
    def build_generator_qcolt_simple(self, input_shape, code_size = 28):

        model = Sequential()

        model.add(InputLayer(input_shape))
        model.add(Flatten())
        model.add(Dense(784,activation='relu'))
        model.add(Dense(50,activation='relu'))
        model.add(Dense(784,activation='relu'))
        model.add(Reshape(input_shape))

        return model
                
        
    def build_generator_qcolt(self, input_shape):
        input_img = Input(shape=input_shape, name='z')
        
        x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)

        encoded = MaxPooling2D((2,2), padding='same')(x)

        # this point representaion is (4,4,8) i.e. 128 dimensional

        x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), activation='relu')(x)
        x = UpSampling2D((2,2))(x)
        
        decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        return autoencoder

    def build_generator_qcolt_deep(self, input_shape):
        input_img = Input(shape=input_shape, name='z')

        #print(input_img.shape)
        #exit()
        
        x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2), padding='same')(x)        
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)

        encoded = MaxPooling2D((2,2), padding='same')(x)

        # this point representaion is (4,4,8) i.e. 128 dimensional
        print(encoded.shape)

        x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2,2))(x)                
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2,2))(x)                
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), activation='relu')(x)
        x = UpSampling2D((2,2))(x)

        #from keras
        
        decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

        #print(decoded.shape)

        #exit()
        autoencoder = Model(input_img, decoded)
        return autoencoder

    def build_generator_qcolt_deep_2(self, input_shape):
        input_img = Input(shape=input_shape, name='z')

        print(input_img.shape)
        
        x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
        x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2,2), padding='same')(x)        
        x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3,3), activation='relu', padding='same')(x)        
        x = MaxPooling2D((2,2), padding='same')(x)        
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)

        encoded = MaxPooling2D((2,2), padding='same')(x)

        # this point representaion is (4,4,8) i.e. 128 dimensional

        x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)        
        x = UpSampling2D((2,2))(x)                
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2,2))(x)                
        x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3,3), activation='relu')(x)
        x = UpSampling2D((2,2))(x)
        #x = Conv2D(16, (3,3), activation='relu')(x)
        #x = UpSampling2D((2,2))(x)

        #from keras
        
        decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

        #print(decoded.shape)

        #exit()
        autoencoder = Model(input_img, decoded)
        return autoencoder

    
    
        
    def build_generator(self, input_shape):
        """Build the generator/R network.
        
        Arguments:
            input_shape {list} -- Generator input shape.
        
        Returns:
            [Tensor] -- Output tensor of the generator/R network.
        """
        image = Input(shape=input_shape, name='z')
        # Encoder.
        x = Conv2D(filters=self.df_dim * 2, kernel_size = 5,
                   strides=2, padding='same', name='g_encoder_h0_conv')(image)
        
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 4, kernel_size = 5,
                   strides=2, padding='same', name='g_encoder_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 8, kernel_size = 5,
                   strides=2, padding='same', name='g_encoder_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decoder.
        # TODO: need a flexable solution to select output_padding and padding.
        # x = Conv2DTranspose(self.gf_dim*2, kernel_size = 5, strides=2, activation='relu', padding='same', output_padding=0, name='g_decoder_h0')(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(self.gf_dim*1, kernel_size = 5, strides=2, activation='relu', padding='same', output_padding=1, name='g_decoder_h1')(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(self.c_dim,    kernel_size = 5, strides=2, activation='tanh', padding='same', output_padding=1, name='g_decoder_h2')(x)

        x = Conv2D(self.gf_dim*1, kernel_size=5,
                   activation='relu', padding='same', name='g_decoded_h3_conv')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim*1, kernel_size=5,
                   activation='relu', padding='same', name='g_decoded_h4_conv')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim*2, kernel_size=3, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.c_dim, kernel_size=5,
                   activation='sigmoid', padding='same', name='g_decoded_h5_conv')(x)
              
        return Model(image, x, name='R')

    
    
    def build_discriminator(self, input_shape):
        """Build the discriminator/D network
        
        Arguments:
            input_shape {list} -- Input tensor shape of the discriminator network, either the real unmodified image
                or the generated image by generator/R network.
        
        Returns:
            [Tensor] -- Network output tensors.
        """

        image = Input(shape=input_shape, name='d_input')
        x = Conv2D(filters=self.df_dim, kernel_size = 5, strides=2, padding='same',
                   name='d_h0_conv')(image)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim*2, kernel_size = 5, strides=2, padding='same',
                   name='d_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim*4, kernel_size = 5, strides=2, padding='same',
                   name='d_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim*8, kernel_size = 5, strides=2, padding='same',
                   name='d_h3_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid',
                  name='d_h3_lin')(x)

        return Model(image, x, name='D')

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        optimizer = RMSprop(lr=0.002, clipvalue=1.0, decay=1e-8)
        # Construct discriminator/D network takes real image as input.
        # D - sigmoid and D_logits -linear output.
        self.discriminator = self.build_discriminator(image_dims)

        # Model to train D to discrimate real images.
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

        # Construct generator/R network.
        #self.generator = self.build_generator(image_dims)
        self.generator = self.build_generator_qcolt(image_dims)
        #self.generator = self.build_generator_qcolt_deep(image_dims)
        #self.generator = self.build_generator_qcolt_deep_2(image_dims)
        #self.generator = self.build_generator_qcolt_simple(image_dims, code_size = 28)

        
        
        img = Input(shape=image_dims)

        reconstructed_img = self.generator(img)

        self.discriminator.trainable = False
        validity = self.discriminator(reconstructed_img)
        
        # Model to train Generator/R to minimize reconstruction loss and trick D to see
        # generated images as real ones.
        self.adversarial_model = Model(img, [reconstructed_img, validity])
        self.adversarial_model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[self.r_alpha, 1],
            optimizer=optimizer)

        print('reconstructor')
        self.generator.summary()
        
        print('discriminator')
        self.discriminator.summary()

        print('adversarial_model')
        self.adversarial_model.summary()

    
    def train(self, epochs, batch_size = 128, sample_interval=500, sigma = 0.155):
        assert self.setup_train_data,"Set Train data before to train"
        
        self.sigma = sigma
        # Make log folder if not exist.
        log_dir = os.path.join(self.log_dir, self.model_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        if self.dataset_name == 'mnist':
            # Get a batch of sample images with attention_label to export as montage.
            sample = self.data[0:batch_size]
        elif self.dataset_name == 'landsat':
            # TODO: Get batch from data
            pass
            
        # Export images as montage, sample_input also use later to generate sample R network outputs during training.
        sample_inputs = np.array(sample).astype(np.float32)
        os.makedirs(self.sample_dir, exist_ok=True)

        cv2.imwrite('./{}/train_input_samples.jpg'.format(self.sample_dir),
                    montage(sample_inputs[:,:,:,0]))
        
        counter = 1
        # Record generator/R network reconstruction training losses.
        plot_epochs = []
        plot_g_recon_losses = []

        # Load traning data, add random noise.
        if self.dataset_name == 'mnist':
            print("Sigma Noise selected to get noisy data = {}".format(sigma))
            sample_w_noise = get_noisy_data(self.data, sigma = sigma)
            
        # Adversarial ground truths
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            print('Epoch ({}/{})-------------------------------------------------'.format(epoch,epochs))
            if self.dataset_name == 'mnist':
                # Number of batches computed by total number of target data / batch size.
                batch_idxs = len(self.data) // batch_size

                
            for idx in range(0, batch_idxs):
                # Get a batch of images and add random noise.
                if self.dataset_name == 'mnist':
                    batch = self.data[idx * batch_size:(idx + 1) * batch_size]
                    batch_noise = sample_w_noise[idx * batch_size:(idx + 1) * batch_size]
                    batch_clean = self.data[idx * batch_size:(idx + 1) * batch_size]
                    
                # Turn batch images data to float32 type.
                batch_images = np.array(batch).astype(np.float32)
                batch_noise_images = np.array(batch_noise).astype(np.float32)
                batch_clean_images = np.array(batch_clean).astype(np.float32)
                
                if self.dataset_name == 'mnist':
                    batch_fake_images = self.generator.predict(batch_noise_images)
                    # Update D network, minimize real images inputs->D-> ones, noisy z->R->D->zeros loss.
                    d_loss_real = self.discriminator.train_on_batch(batch_images, ones)
                    d_loss_fake = self.discriminator.train_on_batch(batch_fake_images, zeros)

                    # Update R network twice, minimize noisy z->R->D->ones and reconstruction loss.
                    self.adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])
                    g_loss = self.adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])    
                    plot_epochs.append(epoch + idx/batch_idxs)
                    plot_g_recon_losses.append(g_loss[1])
                    
                counter += 1
                msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss: {3:>0.3f}, g_loss:{4:>0.3f}, g_recon_loss:{4:>0.3f}'.format(epoch, idx, batch_idxs, d_loss_real+d_loss_fake, g_loss[0], g_loss[1])
                print(msg)
                logging.info(msg)
                if np.mod(counter, sample_interval) == 0:
                    if self.dataset_name == 'mnist':
                        samples    = self.generator.predict(sample_inputs)
                        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                        save_images(samples, [manifold_h, manifold_w],
                            './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))

            # Save the checkpoint end of each epoch.
            self.save(epoch,batch_size,sample_interval)
            #plt.plot(plot_epochs,plot_g_recon_losses)
            #plt.savefig('plot_g_recon_losses_{}.png'.format(epoch))

        # Export the Generator/R network reconstruction losses as a plot.
        plt.title('Generator/R network reconstruction losses')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(plot_epochs,plot_g_recon_losses)
        plt.savefig('plot_g_recon_losses.png')

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.dataset_name,
            self.output_height, self.output_width)

    def save(self, step, batch_size, sample_interval):
        """Helper method to save model weights.
        
        Arguments:
            step {[type]} -- [description]
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        noise = int(self.sigma*10000)
        model_name = 'ALOCC_ep_{}_iz_{}_bs_{}_si_{}_sig_{}.h5'.format(step,
                                                                     self.input_height,
                                                                     batch_size,
                                                                     sample_interval,
                                                                     noise)
        
        self.adversarial_model.save_weights(os.path.join(self.checkpoint_dir, model_name))


if __name__ == '__main__':

    # EXP03
    root = "/QCOLT/QCOLT_DEV_OPS/"
    input_folder = root + '/TDS_NOVELTY_DETECTION/EXP_03/nominal_chips/'
    train_data = load_input_data(input_folder, False) 

    
    model = ALOCC_Model(dataset_name='mnist',
                        input_height=28,
                        input_width=28,
                        train_data=(True,train_data))
    model.set_train_data(train_data)

    #model.train(epochs=10, batch_size=500, sample_interval=9000)

    # EXP03 Dataset 45K (28x28)    
    model.train(epochs=10, batch_size=500, sample_interval=5000)

    # Observations
    # Sigma high >1 or low <0.01 produces discriminator output cloase to 0 in training samples which is bad
    # bathc size 500 and sample Interval 5000 produces good results WORKS!! why???
    
        
    # EXP04 DataSet 120K (28x28) : Small batch size ==> Stochastic Gradient Descent (better generatization)
    # sigma 0.155
    # Results: g_recon_loss stable around 0.5
    #model.train(epochs=10, batch_size=500, sample_interval = 10000, sigma = 0.5) # Ver. AutoEncoder --> qcolt

    # Observations:
    
    # Increasing sigma (to gen noisy data) improves stability (use sigma 0.5)
    # Low batch sizes (50) decrease stability (better use batch size 500)
    # Changing sample interval afects first epocs of the training loss
    # Randomization on input samples produces a lot of inestability

    # EXPERIMENTS
    # dataset  epocs batch  sample sigma  d_loss g_loss g_recon_loss randmized stable
    # -------  ----- -----  ------  ----  -----  ------ ------------ --------- ------ 
    #  exp_04      5   500  50000    0.5  1.395  0.888   0.888       NO        YES    
    #  exp_04      5   500  50000    0.5  0.315  0.5357  5.357       YES       NO
    
    #  exp_04      5   250  10000    0.5  1.395  0.888   0.888       NO

    
    
    #model = ALOCC_Model(dataset_name='landsat', input_height=28,input_width=28)
    #model.train(epochs=5, batch_size=128, sample_interval=500)

    
