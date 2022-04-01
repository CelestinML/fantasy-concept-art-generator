from model import *

train_data_dir = r"C:\Users\Utilisateur\Desktop\UQAC\Sujet_IA\Projet\bdd_dragon\train"
test_data_dir = r"C:\Users\Utilisateur\Desktop\UQAC\Sujet_IA\Projet\bdd_dragon\test"

gan = MachineLearningClassifier(save = "generator_dragon_1.h5", epochs = 5, nb_train_samples=5212, nb_validation_samples=17, class_mode = 'binary', batch_size = 16, img_shape = (35,35,1), train_dir = train_data_dir, test_dir = test_data_dir)
gan.load_data()
print('load ok')
gan.prep_gan()
print('sep ok')
print("données d'entrainement :")
print(len(gan.X))
print('\n')
gan.gan()
print('Etape 1 ok !')
print('\n')
gan.gan_suite(batch_size=50, noise_shape=50)
gan.save_generator()