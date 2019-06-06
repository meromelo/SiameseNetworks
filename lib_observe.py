#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fileencoding=UTF-8

######################
##
##  M.Nakazawa
##

##
##  library
##
import os
from keras.callbacks	import Callback as keras_callbacks_Callback

#
# hyperdash
#
from hyperdash			import Experiment as hyperdash_experiment

"""
exp = hyperdash_experiment('Keras MNIST')
hd_callback = hyperdash_callback(['val_acc','val_loss'], exp)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[hd_callback])
exp.end()
"""


class hyperdash_callback(keras_callbacks_Callback):
    def __init__(self, exp, entries=['val_acc','val_loss']):
        super(hyperdash_callback, self).__init__()
        self.entries    = entries
        self.exp        = exp

    def on_epoch_end(self, epoch, logs=None):
        for entrie in self.entries:
            log = logs.get(entrie)
            if log is not None:
                self.exp.metric(entrie, log)

def get_hyperdash_api_key():
    key_str = os.environ.get('HYPERDASH_API_KEY', default=None)
    return key_str

