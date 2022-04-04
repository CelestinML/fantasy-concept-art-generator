from model import *

train_data_dir = r".\bdd_pokemon\train"
test_data_dir = r".\bdd_pokemon\test"

gan = MachineLearningClassifier(save="pokemon.h5", epochs=2001, nb_train_samples=5212, nb_validation_samples=17,
                                class_mode='binary', batch_size=16, img_shape=(120, 120, 3), train_dir=train_data_dir,
                                test_dir=test_data_dir)
gan.load_data(Preprocess.NONE)
gan.prep_gan()
gan.gan(noise_shape = 100)
gan.gan_suite(batch_size=50, noise_shape=100, save = "pokemon1")
gan.save_generator()
