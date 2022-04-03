from model import *

train_data_dir = r".\bdd_pokemon\train"
test_data_dir = r".\bdd_pokemon\test"

gan = MachineLearningClassifier(save="generator_orc_1.h5", epochs=50, nb_train_samples=5212, nb_validation_samples=17,
                                class_mode='binary', batch_size=16, img_shape=(120, 120, 3), train_dir=train_data_dir,
                                test_dir=test_data_dir)
gan.load_data(Preprocess.NONE)
print('load ok')
gan.prep_gan()
plt.imshow(gan.X_df[0])
plt.show()
print('sep ok')
print("données d'entrainement :")
print(len(gan.X_df))
print("Shape")
print(gan.X_df.shape)
print('\n')
gan.gan(noise_shape = 100)
print('Etape 1 ok !')
print('\n')
gan.gan_suite(batch_size=50, noise_shape=100, save = "pokemon1")
gan.save_generator()
