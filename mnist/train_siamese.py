from siamese_net import SiameseNet
from keras.optimizers import RMSprop
from siamese_data_loader import SiameseDataLoader
import os

if __name__ == '__main__':
    iterations = 10000
    samples_per_class = 5
    feature_dim = 10
    grayscale = True
    # Adam not works well for Siamese net
    optim = RMSprop(decay=1e-4)
    #optim = Adam(lr=0.0001, decay=1e-4, amsgrad=True)

    loader_train = SiameseDataLoader('train' + os.sep, samples_per_class, grayscale)
    loader_val = SiameseDataLoader('val' + os.sep, samples_per_class, grayscale)

    siamese = SiameseNet(loader_train.input_shape, feature_dim).get_model()
    siamese.compile(optimizer=optim, loss=contrastive_loss)
    min_loss = 9999
    min_iter = -1
    for iteration in range(iterations):
        X, y = loader_train.get_train_data()
        loss_train = siamese.train_on_batch(X, y)
        if (iteration+1)%100 == 0:
            X, y = loader_val.get_train_data()
            loss_val = siamese.evaluate(X, y, verbose=0)
            if loss_val < min_loss:
                min_iter = iteration
                min_loss = loss_val
                siamese.save_weights('weights.h5', True)
            print('loss@' + str(iteration) + ' = ' + str(loss_train) + ',' + str(loss_val) + ' (' + str(min_loss) + '@' + str(min_iter) + ')')
