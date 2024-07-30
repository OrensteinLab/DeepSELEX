import numpy as np
import pandas as pd
from keras.utils import to_categorical
import functools
import read_files

class TrainData:
    def __init__(self, selex_str_len, selex_files_num):
        self.linker_sequence_length = 4
        self.selex_str_len = selex_str_len + 2 * self.linker_sequence_length
        self.selex_files_num = selex_files_num
        self.one_hot_data = np.array
        self.enrichment_matrix = np.array

    def set_one_hot_matrix(self, dna_data, primary_selex_sequence):
        start_linker, end_linker = read_files.selex_linker_sequence(file_address='selex_linker_flie.xlsx', primary_selex_sequence=primary_selex_sequence)
        
        self.one_hot_data = np.array(
            list(map(functools.partial(self.one_hot_encoder, start_linker=start_linker, end_linker=end_linker), dna_data)),
            dtype=object
        )

        if start_linker:
            print(f'start linker is: {start_linker[-self.linker_sequence_length:]} and the end linker is: {end_linker[:self.linker_sequence_length]}')
        else:
            self.one_hot_data = self.linker_quarter_padding(modified_matrix=self.one_hot_data)
            print("====Inside create_data::self.one_hot_data:", self.one_hot_data.shape)

    def set_enrichment_matrix(self, enrichment_data):
        self.enrichment_matrix = np.asarray([enrichment_data[k] for k in range(len(enrichment_data))])

    def one_hot_encoder(self, DNA_string, **kwargs):
        if kwargs['start_linker'] is None:
            start_linker = end_linker = "A" * self.linker_sequence_length
        else:
            start_linker = kwargs['start_linker'][-self.linker_sequence_length:]
            end_linker = kwargs['end_linker'][:self.linker_sequence_length]

        DNA_string = start_linker + DNA_string + end_linker + "ACGTN"
        trantab = DNA_string.maketrans('ACGTN', '01234')
        data = list(DNA_string.translate(trantab))
        return to_categorical(data, num_classes=5)[0:-5]

    def linker_quarter_padding(self, modified_matrix):
        if len(modified_matrix.shape) == 3:
            modified_matrix[:, 0:self.linker_sequence_length, :] = 0.25
            modified_matrix[:, self.selex_str_len - self.linker_sequence_length:self.selex_str_len, :] = 0.25
        return modified_matrix

class PredictData:
    def __init__(self, selex_str_len, predict_str_len):
        self.selex_str_len = selex_str_len
        self.predict_str_len = predict_str_len
        self.num_of_str = max(self.predict_str_len - self.selex_str_len - 1, 1)
        self.selex_predict_str_adaptor = int(max((self.selex_str_len - self.predict_str_len) / 2, 0))
        self.one_hot_data = None

    def set_one_hot_matrix(self, dna_data):
        self.one_hot_data = np.array(list(map(self.one_hot_encoder, dna_data)), dtype=object)
        if self.selex_predict_str_adaptor > 0:
            self.one_hot_data = self.set_redundant_linker_to_avergae(modified_matrix=self.one_hot_data)

    def one_hot_encoder(self, DNA_string):
        if self.selex_predict_str_adaptor != 0:
            DNA_string = "A" * self.selex_predict_str_adaptor + DNA_string + 'A' * self.selex_predict_str_adaptor

        trantab = DNA_string.maketrans('ACGTN', '01234')
        str_arr = ["" for x in range(self.num_of_str)]
        for i in range(0, self.num_of_str):
            str_arr[i] = DNA_string[i: i + self.selex_str_len]

        str_arr[self.num_of_str - 1] = str_arr[self.num_of_str - 1] + "ACGTN"

        final_str = list("")
        for i in range(0, self.num_of_str):
            final_str += list(str_arr[i].translate(trantab))

        return to_categorical(final_str, num_classes=5)[0:-5]

    def set_redundant_linker_to_avergae(self, modified_matrix):
        if len(modified_matrix.shape) == 3:
            modified_matrix[:, 0:self.selex_predict_str_adaptor, :] = 0.25
            modified_matrix[:, self.selex_str_len - self.selex_predict_str_adaptor:self.selex_str_len, :] = 0.25
        return modified_matrix

def train_data_constructor(learning_files_list):
    if learning_files_list is None:
        train_data = None
    else:
        full_learning_data_frame = pd.concat(learning_files_list[i].raw_data for i in range(len(learning_files_list)))
        full_learning_data_frame = full_learning_data_frame.sample(frac=1)
        train_data = TrainData(selex_str_len=len(learning_files_list[0].raw_data['DNA_Id'].iloc[0]), selex_files_num=len(learning_files_list))
        train_data.set_one_hot_matrix(dna_data=full_learning_data_frame['DNA_Id'], primary_selex_sequence=learning_files_list[0].primary_selex_sequence)
        train_data.set_enrichment_matrix(enrichment_data=np.asarray(full_learning_data_frame['cycle_matrix']))
    return train_data

def prediction_data_constructor(prediction_file, model_input_size):
    if prediction_file is None:
        prediction_data = None
    else:
        prediction_data = PredictData(selex_str_len=model_input_size, predict_str_len=len(prediction_file.raw_data['DNA_Id'].iloc[0]))
        prediction_data.set_one_hot_matrix(dna_data=prediction_file.raw_data['DNA_Id'])
    return prediction_data
