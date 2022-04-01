import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import cv2
import joblib

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, LeakyReLU, BatchNormalization, \
    Reshape
from keras import backend as K

class PrepaData():
    """ Manage data preparation.

    Parameters
    ------------

    train_generator : generator
        Generator of train data for data augmentation (images databases)

    test_generator : generator
        Generator of train data for data augmentation (images databases)

    val_generator : generator
        Generator of train data for data augmentation (images databases)

    class_mode : string
        String corresponding to the type of problem for image classification 
        ex : 'binary'

    img_shape : array, shape = (width, height, nb_canal)
        Array corresponding to image shape
        nb_canal = 1 if gray image, if rgb image nb_canal = 3

    classes : dict
        Dictionnary with different problem classes 

    batch_size : int 
        Size of batch used during training 

    train_dir : string
        Path to access train data
    
    test_dir : string
        Path to acces test data
    
    val_dir : string
        Path to acces validation data

    X : DataFrame
        DataFrame contenant les données 

    X_normalized : DataFrame
        DataFrame contenant les données normalisées

    y : DataFrame
        Dataframe contenant les labels

    X_train : Dataframe
        DataFrame contenant les données d'entrainement

    X_test : Dataframe
        DataFrame contenant les données de test

    X_val : Dataframe
        DataFrame contenant les données de validation

    y_train : Dataframe
        DataFrame contenant les labels d'entrainement

    y_test : Dataframe
        DataFrame contenant les labels de test

    y_val : Dataframe
        DataFrame contenant les labels de validation

    """

    def __init__(self, class_mode, X=None, y=None, batch_size=None, img_shape=None, train_dir=None, test_dir=None,
                 val_dir=None):
        self.train_generator = None
        self.test_generator = None
        self.val_generator = None
        self.class_mode = class_mode
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.classes = {}
        self.X_df = pd.DataFrame()
        self.X = X
        self.X_normalized = None
        self.y = y
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

    ########################
    ########################
    ########################
    ########################
    ########################
    ## Usefull functions ###
    ########################
    ########################
    ########################
    ########################
    ########################

    def separation_data(self, pourcentage_test=20):
        """Separate data into training and test set.

        Parameters
        ------------
        pourcentage_test : int Percentage for test set.

        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.asarray(self.X), np.asarray(self.y),
                                                                                test_size=pourcentage_test / 100,
                                                                                random_state=42)
        # self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(self.X_test, self.y_test, test_size=0.1, random_state=42)
        return None

    def normalize_data(self):
        """Normalize data to use them to feed neural network"""
        self.X = preprocessing.normalize(self.X)
        return None

    def load_data(self):
        """Load data into X and y"""
        x = []
        y = []
        classes = []

        for i in os.listdir(self.train_dir):
            classes.append(i)
        train_path = []
        test_path = []
        val_path = []

        for i in range(len(classes)):
            self.classes[classes[i]] = i

        if self.val_dir:

            for i in classes:
                chemin = self.train_dir + '/' + i
                train_path.append(chemin)
                chemin = self.test_dir + '/' + i
                test_path.append(chemin)
                chemin = self.val_dir + '/' + i
                val_path.append(chemin)

            for i in range(len(train_path)):
                l_x = []
                for j in os.listdir(test_path[i]):
                    chemin = test_path[i] + '/' + j
                    chemin = chemin.replace('\\', '/')
                    l_x.append(chemin)
                for j in os.listdir(train_path[i]):
                    chemin = train_path[i] + '/' + j
                    chemin = chemin.replace('\\', '/')
                    l_x.append(chemin)
                for j in os.listdir(val_path[i]):
                    chemin = val_path[i] + '/' + j
                    chemin = chemin.replace('\\', '/')
                    l_x.append(chemin)

                for j in range(len(l_x)):
                    img = cv2.imread(l_x[j])
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]
                    img = np.array([cv2.resize(img, (img_width, img_height))])
                    x.append(img)  # car dans le cas de notre problème, on ne veut que des images en niveau de gris
                for j in range(len(l_x)):
                    y.append(i)

            self.X = x
            self.y = np.asarray(y)

        else:
            for i in classes:
                chemin = self.train_dir + '/' + i
                train_path.append(chemin)
                chemin = self.test_dir + '/' + i
                test_path.append(chemin)

            for i in range(len(train_path)):
                l_x = []
                for j in os.listdir(test_path[i]):
                    chemin = test_path[i] + '/' + j
                    chemin = chemin.replace('\\', '/')
                    l_x.append(chemin)
                for j in os.listdir(train_path[i]):
                    chemin = train_path[i] + '/' + j
                    chemin = chemin.replace('\\', '/')
                    l_x.append(chemin)

                for j in range(len(l_x)):
                    img = cv2.imread(l_x[j])
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.bitwise_not(img)  # On inverse les couleurs
                    img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]
                    img = np.array([cv2.resize(img, (img_width, img_height))])
                    x.append(img)  # car dans le cas de notre problème, on ne veut que des images en niveau de gris
                for j in range(len(l_x)):
                    y.append(i)

            self.X = x
            self.y = np.asarray(y)

        return None

    ########################
    ########################
    ########################
    ########################
    ########################
    ######### GAN ##########
    ########################
    ########################
    ########################
    ########################
    ########################

    def prep_gan(self):
        """Data preparation for GAN use
        1 - Data normalization
        2 - Store data into a dataframe
        """
        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]

        if K.image_data_format() == 'channels_first':
            self.input_shape = (nb_canaux, img_width, img_height)
        else:
            self.input_shape = (img_width, img_height, nb_canaux)
        for i in range(len(self.X)):
            self.X_df.append(pd.Series(self.X[i].flatten()), ignore_index=True)
        self.X_df = self.X_df.values
        self.X_df = self.X_df.reshape(-1, img_width, img_height, nb_canaux)
        self.X_df = self.X_df / 255


# ----------------------------------------------------------------------------------------------------# 100 -
# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------#
# --------------------------------------------OTHER  CLASS--------------------------------------------#
# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------#


class MachineLearningClassifier(PrepaData):
    """ Manage data preparation.

    Parameters
    ------------

    nb_train_samples : int
        Number of samples during training
    
    save : string
        Path where you want to save the train model
    
    nb_validation_samples : int
        Number of samples for the validation

    confusion : DataFrame
        Dataframe corresponding to the confusion matrix
    
    input_shape : array, shape = (width, height, nb_canal) ou shape = (nb_canal, width, height)

    history : ??? 
        Informations about model training

    generator : model
        Generator model for GAN

    discriminator : model
        Discrominator model for GAN

    GAN : model
        GAN model

    """

    def __init__(self, epochs, nb_train_samples, nb_validation_samples, class_mode, save=None, X=None, y=None,
                 batch_size=None, img_shape=None, train_dir=None, test_dir=None, val_dir=None):
        super().__init__(class_mode, X, y, batch_size, img_shape, train_dir, test_dir, val_dir)
        self.model = None
        self.epochs = epochs
        self.save = save
        self.confusion = None
        self.nb_validation_samples = nb_validation_samples
        self.nb_train_samples = nb_train_samples
        self.history = None
        self.input_shape = None
        self.generator = None
        self.discriminator = None
        self.GAN = None

    ########################
    ########################
    ########################
    ########################
    ########################
    ## Usefull functions ###
    ########################
    ########################
    ########################
    ########################
    ########################

    def load_model(self, nom):
        """"Permit to load an existent model into our class object
        
        Parameter
        ------------
        nom : str Path to model place

        """
        try:
            self.model = joblib.load(nom)

        except:
            self.model = keras.models.load_model(nom)

    def save_model(self):
        """Save the model in order to use it later"""
        try:
            self.model.save(self.save)
        except:
            joblib.dump(self.model, self.save)

    ########################
    ########################
    ########################
    ########################
    ########################
    ######### GAN ##########
    ########################
    ########################
    ########################
    ########################
    ########################

    def gan(self):
        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]

        if K.image_data_format() == 'channels_first':
            self.input_shape = (nb_canaux, img_width, img_height)
        else:
            self.input_shape = (img_width, img_height, nb_canaux)

        self.generator = Sequential()
        self.generator.add(Dense(512, input_shape=[100]))
        self.generator.add(LeakyReLU(alpha=0.2))
        self.generator.add(BatchNormalization(momentum=0.8))
        self.generator.add(Dense(256))
        self.generator.add(LeakyReLU(alpha=0.2))
        self.generator.add(BatchNormalization(momentum=0.8))
        self.generator.add(Dense(128))
        self.generator.add(LeakyReLU(alpha=0.2))
        self.generator.add(BatchNormalization(momentum=0.8))
        self.generator.add(Dense(img_width * img_height))
        self.generator.add(Reshape([img_width, img_height, nb_canaux]))

        self.discriminator = Sequential()
        self.discriminator.add(Dense(1, input_shape=[img_width, img_height, 1, nb_canaux]))
        self.discriminator.add(Flatten())
        self.discriminator.add(Dense(256))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(0.5))
        self.discriminator.add(Dense(128))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(0.5))
        self.discriminator.add(Dense(64))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(0.5))
        self.discriminator.add(Dense(1, activation='sigmoid'))

        self.GAN = Sequential([self.generator, self.discriminator])
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy')
        self.discriminator.trainable = False

        self.GAN.compile(optimizer='adam', loss='binary_crossentropy')

    def gan_suite(self, batch_size, noise_shape):
        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]

        if K.image_data_format() == 'channels_first':
            self.input_shape = (nb_canaux, img_width, img_height)
        else:
            self.input_shape = (img_width, img_height, nb_canaux)

        epochs = self.epochs
        batch_size = batch_size
        noise_shape = noise_shape

        for epoch in range(epochs):
            print(f"Currently on Epoch {epoch + 1}")

            for i in range(self.X_df.shape[0] // batch_size):

                if (i + 1) % 50 == 0:
                    print(f"\tCurrently on batch number {i + 1} of {self.X_df.shape[0] // batch_size}")

                noise = np.random.normal(size=[batch_size, noise_shape])

                gen_image = self.generator.predict_on_batch(noise)

                train_dataset = self.X_df[i * batch_size:(i + 1) * batch_size]

                # training discriminator on real images
                train_label = np.ones(shape=(batch_size, 1))
                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(train_dataset, train_label)

                # training discriminator on fake images
                train_label = np.zeros(shape=(batch_size, 1))
                d_loss_fake = self.discriminator.train_on_batch(gen_image, train_label)

                # training generator
                noise = np.random.normal(size=[batch_size, noise_shape])
                train_label = np.ones(shape=(batch_size, 1))
                self.discriminator.trainable = False

                d_g_loss_batch = self.GAN.train_on_batch(noise, train_label)

            # plotting generated images at the start and then after every 4 epoch
            # if epoch % 10 == 0:
            if epoch % 4 == 0:
                samples = 10
                x_fake = self.generator.predict(np.random.normal(loc=0, scale=1, size=(samples, 100)))

                for k in range(samples):
                    plt.subplot(2, 5, k + 1)
                    plt.imshow(x_fake[k].reshape(img_width, img_height), cmap='gray')
                    plt.xticks([])
                    plt.yticks([])

                plt.tight_layout()
                plt.show()

        print('Training is complete')

    def save_generator(self):
        """Save the model in order to use it later"""
        self.generator.save(self.save)
