from siamese_net import SiameseNet, contrastive_loss
import numpy as np
from keras.optimizers import RMSprop
from siamese_data_loader import SiameseDataLoader
import os

# Siamese Networkはペア画像との距離を返してくるので、
# 一番近い距離の画像が所属しているクラスを予測クラスとする
def distance_to_class(y, classes, samples_per_class):
    i = 0
    class_distances = []
    for c in range(classes):
        distances = []
        for s in range(samples_per_class):
            distances.append(y[i])
            i += 1
        median = np.median(np.array(distances))
        class_distances.append(median)
    return np.argmin(np.array(class_distances))

if __name__ == '__main__':
    samples_per_class = 5
    feature_dim = 10
    grayscale = True
    optim = RMSprop(decay=1e-4)
    test_root_path = 'test' + os.sep
    classes = 10

    loader = SiameseDataLoader('train' + os.sep, samples_per_class, grayscale)

    siamese = SiameseNet(loader.input_shape, feature_dim).get_model()
    siamese.compile(optimizer=optim, loss=contrastive_loss)
    siamese.load_weights('weights.h5')

    correct = 0
    count = 0
    for c in range(classes):
        test_class_folder_path = test_root_path + str(c) + os.sep
        test_file_names = os.listdir(test_class_folder_path)
        distances = []
        for test_file_name in test_file_names:
            test_file_path = test_class_folder_path + test_file_name
            X = loader.get_test_data(test_file_path, samples_per_class)
            y = siamese.predict_on_batch(X)
            predicted_class = distance_to_class(y, classes, samples_per_class)
            if predicted_class == c:
                correct += 1
            count += 1
    accuracy = correct/count*100
    print('accuracy=' + str(accuracy))
