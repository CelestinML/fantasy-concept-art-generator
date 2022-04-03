from model import *

train_data_dir = r".\bdd_dragon\train"
test_data_dir = r".\bdd_dragon\test"

gan = MachineLearningClassifier(save="generator_dragon_1.h5", epochs=101, nb_train_samples=5212, nb_validation_samples=17,
                                class_mode='binary', batch_size=16, img_shape=(35, 35, 1), train_dir=train_data_dir,
                                test_dir=test_data_dir)
gan.load_data(Preprocess.OUTLINES)
print('load ok')
gan.prep_gan()
print('sep ok')
print("données d'entrainement :")
print(gan.X_df)
print("Shape")
print(gan.X_df.shape)
print('\n')
gan.gan()
print('Etape 1 ok !')
print('\n')
gan.gan_suite(batch_size=50, noise_shape=100)
gan.save_generator()
