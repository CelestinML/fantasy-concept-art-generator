from model import *

train_data_dir = r".\bdd_pokemon\train"
test_data_dir = r".\bdd_pokemon\test"

gan = MachineLearningClassifier(save="pokemons4.h5", epochs=1001, nb_train_samples=5212, nb_validation_samples=17,
                                class_mode='binary', batch_size=16, img_shape=(120, 120, 1), train_dir=train_data_dir,
                                test_dir=test_data_dir)
gan.load_data([Preprocess.NONE, Preprocess.HORIZONTAL_FLIP])
gan.prep_gan()
print("Nb images : " + str(len(gan.X_df)))
gan.gan_conv(noise_shape = 100)
gan.gan_suite(batch_size=50, noise_shape=100, save = "pokemons4")
gan.save_generator()