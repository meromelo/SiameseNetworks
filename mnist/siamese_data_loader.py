import os, random, cv2
import numpy as np

class SiameseDataLoader(object):
    def __init__(self, root_train_folder_path, samples_per_class, grayscale=False):
        self._root_train_folder_path = root_train_folder_path
        self._samples_per_class = samples_per_class
        self._sample_file_names = self._get_samples()
        self._grayscale = grayscale
        if self._grayscale:
            image = cv2.imread(self._sample_file_names[0][0], cv2.IMREAD_GRAYSCALE)
            self.input_shape = (image.shape[0], image.shape[1], 1)
        else:
            image = cv2.imread(self._sample_file_names[0][0])
            self.input_shape = image.shape

    def get_train_data(self):
        # positiveとnegativeの画像ペアファイルパスを受け取る
        pairs, labels = self._create_pairs(self._sample_file_names, self._samples_per_class)
        tmp = cv2.imread(pairs[0][0])
        if self._grayscale:
            X1 = np.zeros((len(pairs), tmp.shape[0], tmp.shape[1], 1), np.float32)
            X2 = np.zeros((len(pairs), tmp.shape[0], tmp.shape[1], 1), np.float32)
        else:
            X1 = np.zeros((len(pairs), tmp.shape[0], tmp.shape[1], tmp.shape[2]), np.float32)
            X2 = np.zeros((len(pairs), tmp.shape[0], tmp.shape[1], tmp.shape[2]), np.float32)
        Y = np.zeros((len(pairs), 1), dtype=np.float32)
        i = 0
        if self._grayscale:
            for pair, label in zip(pairs, labels):
                x1 = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
                X1[i] = x1[:,:,np.newaxis]
                x2 = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
                X2[i] = x2[:,:,np.newaxis]
                Y[i] = labels[i]
                i += 1
        else:
            for pair, label in zip(pairs, labels):
                X1[i] = cv2.imread(pair[0])
                X2[i] = cv2.imread(pair[1])
                Y[i] = labels[i]
                i += 1
        return [self._normalize(X1), self._normalize(X2)], Y

    def _get_samples(self):
        sample_file_names = []
        folders = os.listdir(self._root_train_folder_path)
        for folder_name in folders:
            folder_path = self._root_train_folder_path + folder_name
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                sample_file_names_per_class = []
                for file in files:
                    sample_file_names_per_class.append(folder_path + os.sep + file)
                sample_file_names.append(sample_file_names_per_class)
        return sample_file_names

    def _create_pairs(self, sample_file_names, samples_per_class):
        positive_pairs, positive_labels = self._create_positive_pairs(sample_file_names, samples_per_class)
        negative_pairs, negative_labels = self._create_negative_pairs(sample_file_names, samples_per_class)
        positive_pairs.extend(negative_pairs)
        positive_labels.extend(negative_labels)

        return positive_pairs, positive_labels

    # 手書き数字の0と0等同じクラスのペアを作成するためのメソッド
    def _create_positive_pairs(self, sample_file_names, samples_per_class):
        positive_pairs = []
        for sample_file_names_per_class in sample_file_names:
            for k in range(samples_per_class):
                positive_pairs.append(random.sample(sample_file_names_per_class, 2))
        labels = [1]*len(positive_pairs)
        return positive_pairs, labels

    # 手書き数字の2と3等異なるクラスのペアを作成するためのメソッド
    def _create_negative_pairs(self, sample_file_names, samples_per_class):
        negative_pairs = []
        class_count = len(sample_file_names)
        for i, sample_file_names_per_class in enumerate(sample_file_names):
            class_ids = list(range(class_count))
            class_ids.remove(i)
            for k in range(samples_per_class):
                pair = []
                pair.append(random.choice(sample_file_names[i]))
                pair.append(random.choice(sample_file_names[random.choice(class_ids)]))
                negative_pairs.append(pair)
        labels = [0]*len(negative_pairs)
        return negative_pairs, labels

    def _normalize(self, X):
        return X/255

    def get_test_data(self, test_image_path, samples_per_class):
        pairs = []
        for sample_file_names_per_class in self._sample_file_names:
            selected_files = random.sample(sample_file_names_per_class, samples_per_class)
            for selected_file in selected_files:
                pair = []
                pair.append(test_image_path)
                pair.append(selected_file)
                pairs.append(pair)
        tmp = cv2.imread(pairs[0][0])
        if self._grayscale:
            X1 = np.zeros((len(pairs), tmp.shape[0], tmp.shape[1], 1), np.float32)
            X2 = np.zeros((len(pairs), tmp.shape[0], tmp.shape[1], 1), np.float32)
            for i, pair in enumerate(pairs):
                X1[i] = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
                X2[i] = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
        else:
            X1 = np.zeros((len(pairs), tmp.shape[0], tmp.shape[1], tmp.shape[2]), np.float32)
            X2 = np.zeros((len(pairs), tmp.shape[0], tmp.shape[1], tmp.shape[2]), np.float32)
            for i, pair in enumerate(pairs):
                X1[i] = cv2.imread(pair[0])
                X2[i] = cv2.imread(pair[1])
        return [self._normalize(X1), self._normalize(X2)]
