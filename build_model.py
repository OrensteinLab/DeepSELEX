import numpy as np
import keras
from keras.layers import *
from keras.callbacks import *
from keras.regularizers import *
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def manage_model(cmd_args, train_data):
    if cmd_args.loaded_model_location:
        return load_model(cmd_args.loaded_model_location)
    elif cmd_args.saved_model_location:
        # Update input_shape to have 5 channels instead of 4
        model = Model(model_path=cmd_args.saved_model_location, input_shape=(train_data.selex_str_len, 5),
                      output_size=train_data.selex_files_num)
        model.create_model()
        model.model_train_and_save(data=train_data)
        return model.model

class Model:
    def __init__(self, model_path, input_shape, output_size):
        self.model = Sequential()
        self.model_path = model_path
        self.input_shape = input_shape
        self.output_size = output_size
        self.model_params_dict = {'ker_size': 8, 'pool_size': 5, 'layers': [64, 32, 32],
                                  'final_activation_function': 'sigmoid'}

    def create_model(self):
        self.model.add(
            Conv1D(filters=512, kernel_size=self.model_params_dict['ker_size'], strides=1,
                   kernel_initializer='RandomNormal',
                   activation='relu',
                   kernel_regularizer=l2(5e-3), input_shape=self.input_shape, use_bias=True,
                   bias_initializer='RandomNormal'))
        self.model.add(MaxPooling1D(pool_size=self.model_params_dict['pool_size'], strides=None,
                               padding='valid',
                               data_format='channels_last'))
        self.model.add(Flatten())
        for layer_size in self.model_params_dict['layers']:
            print(f'the layer size is: {layer_size}')
            self.model.add(Dense(layer_size, activation='relu'))
        print("======> output_size", self.output_size)
        self.model.add(Dense(self.output_size, activation=self.model_params_dict['final_activation_function']))

    def model_train_and_save(self, data):
        Adam = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False)
        self.model.compile(optimizer=Adam, loss='categorical_crossentropy')
        datasets = 3
        splitted_train_data, splitted_test_data = self.data_divider(data=data, datasets=datasets)
        for i in range(datasets):
            self.model.fit(splitted_train_data[i], splitted_test_data[i], epochs=30,
                        batch_size=64, verbose=1, shuffle=True, validation_split=0.3,
                        callbacks=[keras.callbacks.ModelCheckpoint(self.model_path, monitor='val_loss', verbose=0, save_best_only=True,
                                                                   save_weights_only=False, mode='auto', period=1),
                                   keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                 min_delta=0,
                                                                 patience=1,
                                                                 verbose=0, mode='auto', restore_best_weights=True)
                                   ])

    def reshape_one_hot_data(self, data):
        print("=====> Inside reshape one hot data..")
        target_shape = (data.selex_str_len, 5)  # Update to match the new target shape
        if isinstance(data.one_hot_data, np.ndarray) and data.one_hot_data.dtype == object:
            # Pad sequences to the target length
            padded_sequences = pad_sequences(data.one_hot_data, maxlen=target_shape[0], padding='post', truncating='post', dtype='float32')
            data.one_hot_data = np.array(padded_sequences)
        else:
            raise ValueError("data.one_hot_data is not in the expected format.")
        return data.one_hot_data

    def data_divider(self, data, datasets=3):
        print("BEFORE=======", data.one_hot_data.shape)
        print("BEFORE=======", data.one_hot_data[0].shape)
        if len(data.one_hot_data.shape) != 3:
            data.one_hot_data = self.reshape_one_hot_data(data)

        splitted_train_data = []
        splitted_test_data = []
        for i in range(0, datasets):
            print("=======", data.one_hot_data.shape)
            print("=======", data.one_hot_data[0].shape)
            print('the ranges in {i} are: {low}, {high}'.format(i=i, low=round((i/datasets)*len(data.one_hot_data)), high=round(((i+1)/datasets) * len(data.one_hot_data))))
            splitted_train_data.append(data.one_hot_data[round((i/datasets)*len(data.one_hot_data)):round(((i+1)/datasets) * len(data.one_hot_data)), :, :])
            splitted_test_data.append(data.enrichment_matrix[round((i/datasets)*len(data.enrichment_matrix)):round(((i+1)/datasets) * len(data.enrichment_matrix))])

        return splitted_train_data, splitted_test_data
