import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import cv2
import sys

from enum import Enum


class Preprocess(Enum):
    NONE = 1
    LEFT_ROTATE = 2
    RIGHT_ROTATE = 3
    ROTATE_180 = 4
    HORIZONTAL_FLIP = 5
    VERTICAL_FLIP = 6


BATCH_SIZE = 128
IMAGE_SIZE = 120
IMAGE_CHANNELS = 1  # can be 3 (RGB) or 1 (Grayscale)
LATENT_SPACE_DIM = 100  # dimensions of the latent space that is used to generate the images

assert IMAGE_SIZE % 4 == 0

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

num_examples_to_generate = 16
# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, LATENT_SPACE_DIM])


def preprocess(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # load the image as uint8 array and transform to grayscale
    img = tf.image.decode_jpeg(img, channels=IMAGE_CHANNELS)
    # resize the image to the desired size
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    # transform the color values from [0, 255] to [-1, 1]. The division changes the datatype to float32
    img = (img - 127.5) / 127.5
    return img


def filter(img):
    return img[0, 0, 0] == -1  # discard white bg images (estimate by the R channel of the top left pixel)


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.filter(filter)
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def make_generator_model():
    model = tf.keras.Sequential()

    n = IMAGE_SIZE // 4

    model.add(layers.Dense(n * n * 256, use_bias=False, input_shape=(LATENT_SPACE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((n, n, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(IMAGE_CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                     activation='tanh'))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                      input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


discriminator = make_discriminator_model()
generator = make_generator_model()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_SPACE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise,
                                     training=True)  # training=True is important, sicne Dropout and BatchNorm behave differently during inference

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs, save_after, model_name):
    generator.save(r"./modeles/" + model_name)
    generate_and_save_images(generator,
                             0,
                             seed)
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

        if (epoch + 1) % save_after == 0:
            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)


def generate_and_save_images(model, epoch, test_input, show=False, save=True):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10, 10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        if predictions.shape[-1] == 3:
            plt.imshow(
                predictions[i] * 0.5 + .5)  # scale image to [0, 1] floats (or you could also scale to [0, 255] ints)
        else:
            plt.imshow(predictions[i, :, :, 0] * 0.5 + .5,
                       cmap='gray')  # scale image to [0, 1] floats (or you could also scale to [0, 255] ints)
        plt.axis('off')
    output_folder = './images/'

    if save:
        plt.savefig(output_folder + 'image_at_epoch_{:04d}.png'.format(epoch))

    if show:
        plt.show()


def apply_preprocess(dataset_path, preprocesses):
    # create the temporary folder if not exist
    temp_dir_path = './temp'
    if not os.path.exists(temp_dir_path):
        os.mkdir(temp_dir_path)
    else:
        for f in os.listdir(temp_dir_path):
            os.remove(os.path.join(temp_dir_path, f))

    imageNames = os.listdir(dataset_path)
    print("Nb images avant preprocess : " + str(len(imageNames)))

    for imageName in imageNames:
        img = cv2.imread(dataset_path + '/' + imageName, cv2.IMREAD_UNCHANGED)
        # resize image
        #img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        # save the image
        cv2.imwrite(temp_dir_path + '/' + imageName, img)

        for preprocess in preprocesses:
            if preprocess == Preprocess.RIGHT_ROTATE:
                image_center = tuple(np.array(img.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, -90, 1.0)
                modified_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            elif preprocess == Preprocess.LEFT_ROTATE:
                image_center = tuple(np.array(img.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, 90, 1.0)
                modified_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            elif preprocess == Preprocess.ROTATE_180:
                image_center = tuple(np.array(img.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, 180, 1.0)
                modified_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            elif preprocess == Preprocess.HORIZONTAL_FLIP:
                modified_img = cv2.flip(img, 1)
            elif preprocess == Preprocess.VERTICAL_FLIP:
                modified_img = cv2.flip(img, 0)
            # save the modified img
            cv2.imwrite(temp_dir_path + '/' + str(preprocess) + '-' + imageName, modified_img)

    return temp_dir_path


def train_model(dataset_path, model_name, epochs, save_after, preprocesses=[]):
    temp_dir_path = apply_preprocess(dataset_path, preprocesses)

    list_ds = tf.data.Dataset.list_files(str(temp_dir_path + '/*'), shuffle=True)  # Get all images
    train_dataset = list_ds.take(-1)
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_dataset = train_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    print("Nb images après preprocess : " + str(len(train_dataset)))
    train_dataset = configure_for_performance(train_dataset)

    train(train_dataset, epochs, save_after, model_name)


def test_model(model_path):
    model = tf.keras.models.load_model(model_path)

    generate_and_save_images(model, -1, seed, True, False)


train_model(dataset_path='./manga_faces', model_name='test_manga1', epochs=10000, save_after=100, preprocesses=[Preprocess.HORIZONTAL_FLIP])
# test_model("./modeles/test_pokemon3")
