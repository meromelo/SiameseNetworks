from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda, Conv2D, Activation, MaxPool2D, BatchNormalization, Dropout, Flatten
import keras.backend as K

class SiameseNet(object):
    def __init__(self, input_shape, feature_dim):
        seq = Sequential()
        seq.add(Conv2D(16, 3, padding='same', input_shape=input_shape))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))
        seq.add(MaxPool2D())

        seq.add(Conv2D(32, 3, padding='same'))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))
        seq.add(MaxPool2D())

        seq.add(Conv2D(64, 3, padding='same'))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))
        seq.add(MaxPool2D())

        seq.add(Flatten())
        seq.add(Dense(256, activation='sigmoid'))
        seq.add(Dropout(0.2))
        seq.add(Dense(feature_dim, activation='linear'))

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        processed_a = seq(input_a)
        processed_b = seq(input_b)
        distance = Lambda(self._euclidean_distance, output_shape=self._eucl_dist_output_shape)([processed_a, processed_b])
        self._model = Model(inputs=[input_a, input_b], outputs=distance)

    def _euclidean_distance(self, vects):
        x, y = vects
        distance = K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
        return distance

    def _eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def get_model(self):
        return self._model

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true*K.square(y_pred) + (1 - y_true)*K.square(K.maximum(margin - y_pred, 0)))
