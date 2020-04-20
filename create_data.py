import numpy as np
import pandas as pd
from keras.utils import to_categorical
import functools
import read_files

class TrainData:
    """
    A class used to build the training data for the DeepSELEX model.

    Attributes
    ----------
    linker_sequence_length : int
        The length of the linker sequences which will be add to the HT-SELEX data
    selex_str_len : int
        The HT-SELEX sequence length plus 2*linker_sequence_length
    selex_files_num : int
        The nuber of the HT-SELEX file number
    one_hot_data : np.array
        This is the input in the learning process. The data needs to be OneHot encoded since this is a CNN DL model
    enrichment_matrix : np.array
        This is the model label
    Methods
    -------
    set_one_hot_matrix()
        Create the One hot matrix
    set_enrichment_matrix()
        Format the enrichment matrix in a way that will enable the matrix to be the model label
    one_hot_encoder(DNA_string, **kwargs)
        Encode each sequence plus linker sequences to OneHot matrices
    """

    def __init__(self, selex_str_len, selex_files_num):
        """
        Attributes
        ----------
        linker_sequence_length : int
            The length of the linker sequences which will be add to the HT-SELEX data
        selex_str_len : int
            The HT-SELEX sequence length plus 2*linker_sequence_length
        selex_files_num : int
            The nuber of the HT-SELEX file number
        one_hot_data : np.array
            This is the input in the learning process. The data needs to be OneHot encoded since this is a CNN DL model
        enrichment_matrix : np.array
            This is the model label"""
        self.linker_sequence_length = 4
        self.selex_str_len = selex_str_len + 2 * self.linker_sequence_length
        self.selex_files_num = selex_files_num
        self.one_hot_data = np.array
        self.enrichment_matrix = np.array

    def set_one_hot_matrix(self, dna_data, primary_selex_sequence):
        """Create the OneHot matrix in two steps.
        1. first fetch the two linker sequence
        2. construct the OneHot matrix with the appropriate linker size.


        Parameters
        ----------
        dna_data : DataFrame
            This is the DNA sequences from the HT-SELEX experiment
        primary_selex_sequence : str
        the sequence which is the HT-SELEX experiment primary sequence.
        If the selex file is of the form: ALX4_TGTGTC20NGA_W_0.fastq, the primary sequence is: TGTGTC20NGA
        """

        start_linker, end_linker = read_files.selex_linker_sequence(file_address='selex_linker_flie.xlsx',
                                                                    primary_selex_sequence=primary_selex_sequence)


        self.one_hot_data = np.array(
            list(map(functools.partial(self.one_hot_encoder, start_linker=start_linker,
                                       end_linker=end_linker), dna_data)))

        if start_linker:
            print(f'start linker is: {start_linker[-self.linker_sequence_length:]}'
                  f' and the end linker is: {end_linker[:self.linker_sequence_length]}')
        else:
            self.one_hot_data = self.linker_quarter_padding(modified_matrix=self.one_hot_data)

    def set_enrichment_matrix(self, enrichment_data):
        """Create the enrichment matrix to suit as the model label.

        Parameters
        ----------
        enrichment_data : DataFrame
            The enrichment data which is supplied from all the learning files
        """
        self.enrichment_matrix = np.asarray([enrichment_data[k] for k in range(len(enrichment_data))])

    def one_hot_encoder(self, DNA_string, **kwargs):
        """Encode each sequence to OneHot matrix
        1. add linker sequence to the beginning and end of the main HT-SELEX sequence
        2. translate the letters to numbers
        3. list them and send back since the matrix is formed by "map" method. if the "ACGT"
        won't be added it will be impossible to convert sequnces which miss one of the letters

        Parameters
        ----------
        DNA_string: str
            This is the DNA sequence. Sent one at a time
        **kwargs: dict
            contains the start and end linker sequences
        """
        if kwargs['start_linker'] is None:
            start_linker = end_linker = "A" * self.linker_sequence_length
        else:
            start_linker = kwargs['start_linker'][-self.linker_sequence_length:]
            end_linker = kwargs['end_linker'][:self.linker_sequence_length]

        # if the "ACGT"
        # won't be added it will be impossible to convert sequnces which miss one of the letters
        DNA_string = start_linker + DNA_string + end_linker + "ACGT"
        trantab = DNA_string.maketrans('ACGT', '0123')
        data = list(DNA_string.translate(trantab))
        return to_categorical(data)[0:-4]  # returns the matrix without the "ACGT"

    def linker_quarter_padding(self, modified_matrix):
        """If the user did not supplied primary_selex_sequence please note that we added:
        "A" * self.linker_sequence_length at the one_hot_encoder
        This need to be replaced by 0.25
        Here we just replace all the places where we putted A, which gave something like this:
        1 1 0 0 1 0 1 1
        0 0 1 0 0 0 0 0
        0 0 0 1 0 0 0 0
        0 0 0 0 0 1 0 0
        to:
        0.25 0.25 0 0 1 0 0.25 0.25
        0.25 0.25 1 0 0 0 0.25 0.25
        0.25 0.25 0 1 0 0 0.25 0.25
        0.25 0.25 0 0 0 1 0.25 0.25

        We repeat this process
        Parameters
        ----------
        modified_matrix: np.array
            This is the matrix we are going to convert
        """
        modified_matrix[:, 0:self.linker_sequence_length, :] = 0.25
        modified_matrix[:, self.selex_str_len - self.linker_sequence_length:self.selex_str_len, :] = 0.25
        return modified_matrix

class PredictData:
    """
    A class used to build the predict data for the DeepSELEX model prediction process.

    Attributes
    ----------
    selex_str_len : int
        The HT-SELEX sequence length plus 2*linker_sequence_length
    predict_str_len : int
        The prediction sequence length
    num_of_str : int
        Those are the number of strings the prediction file will be divided into.
        if the prediction file sequences  longer than the HT-SELEX sequences (plus 2 times linker) we will divide
        the predicition file sequences into smaller and selex_str_len sequences so we could use them as input to
        the model in the prediction process.
    selex_predict_str_adaptor : int
        If the HT-SELEX sequence (plus 2 times linker) are longer from the prediction file sequences we need
        to bridge those sequences since the model input will always be the HT-SELEX sequence length.
        This means we will just padd the prediction edges in 0.25
    one_hot_data : np.array
        This is the input in the prediction process. The data needs to be OneHot encoded since this is a CNN DL model

    Methods
    -------
    set_one_hot_matrix()
        Create the One hot matrix
    one_hot_encoder(DNA_string, **kwargs)
        Encode each sequence plus linker sequences to OneHot matrices
    selex_predict_str_adaptor(DNA_string, **kwargs)
        If we need to bridge the prediction sequences to the HT-SELEX sequences we will use this function
    """
    def __init__(self, selex_str_len, predict_str_len):
        self.selex_str_len = selex_str_len
        self.predict_str_len = predict_str_len
        self.num_of_str = max(self.predict_str_len - self.selex_str_len - 1, 1)
        self.selex_predict_str_adaptor = int(max((self.selex_str_len - self.predict_str_len) / 2, 0))
        self.one_hot_data = None

    def set_one_hot_matrix(self, dna_data):

        """this method is responsible to produce the OneHot
        encoding for the DNA strings.
        The OneHOt matrix is used for the convolutional process
        If we need to bridge between the prediction to HT-SELEX sequences we will use
        set_redundant_linker_to_avergae method
        Parameters
        ----------
        dna_data : DataFrame
            This is the DNA sequences from the HT-SELEX experiment"""

        self.one_hot_data = np.array(list(map(functools.partial(self.one_hot_encoder), dna_data)))
        if self.selex_predict_str_adaptor > 0:
            self.one_hot_data = self.set_redundant_linker_to_avergae(modified_matrix=self.one_hot_data)


    def one_hot_encoder(self, DNA_string):
        """Encode each sequence to OneHot matrix
        1. Add "A" in case we need to bridge prediction to HT-SELEX
        2. Translate the letters to numbers
        3. list them and send back since the matrix is formed by "map" method. if the "ACGT"
        won't be added it will be impossible to convert sequnces which miss one of the letters

        Parameters
        ----------
        DNA_string: str
            This is the DNA sequence. Sent one at a time
        """

        if self.selex_predict_str_adaptor != 0:
            string = "A" * self.selex_predict_str_adaptor + DNA_string + 'A' * self.selex_predict_str_adaptor

        trantab = DNA_string.maketrans('ACGT', '0123')
        str_arr = ["" for x in range(self.num_of_str)]
        for i in range(0, self.num_of_str):  ##each substring goes to different element array
            str_arr[i] = DNA_string[i: i + self.selex_str_len]

        # if the "ACGT"
        # won't be added it will be impossible to convert sequnces which miss one of the letters
        str_arr[self.num_of_str - 1] = str_arr[self.num_of_str - 1] + "ACGT"

        final_str = list("")
        for i in range(0, self.num_of_str):
            final_str += list(str_arr[i].translate(trantab))

        return to_categorical(final_str)[0:-4]  # returns the matrix without the "ACGT"

    def set_redundant_linker_to_avergae(self, modified_matrix):
        """If we need to bridge between the prediction file sequences to the HT-SELEX sequences
        we use this function. In the one_hot_encoder you can see we add "A" if we need to bridge.
        Here we just replace all the places where we putted A, which gave something like this:
        1 1 0 0 1 0 1 1
        0 0 1 0 0 0 0 0
        0 0 0 1 0 0 0 0
        0 0 0 0 0 1 0 0
        to:
        0.25 0.25 0 0 1 0 0.25 0.25
        0.25 0.25 1 0 0 0 0.25 0.25
        0.25 0.25 0 1 0 0 0.25 0.25
        0.25 0.25 0 0 0 1 0.25 0.25

        We repeat this process
        Parameters
        ----------
        modified_matrix: np.array
            This is the matrix we are going to convert
        """
        modified_matrix[:, 0:self.selex_predict_str_adaptor, :] = 0.25
        modified_matrix[:, self.selex_str_len - self.selex_predict_str_adaptor:self.selex_str_len, :] = 0.25
        return modified_matrix



def train_data_constructor(learning_files_list):
    """This function will construct TrainData object.
     The process is as follows
     In the first step it will just __init__ the
    class and afterwards:
     1. builds full_learning_data_frame which is a DataFrame which is concatenated from all the learning_files_list
     raw_data
     2. Shuffle the DataFrame
     3. __init__ the TrainData class
     4. Add a OneHot matrix constructed from all the learning files
     5. Add an enrichment matrix constructed from all the learning files to class

    :parameter
     - `prediction_file`: PredictionFile object
     - `train_data`: TrainData object which is necessery to build the PredictData class
    :returns
     - `train_data`: TrainData object which will be used in the training process"""

    if learning_files_list is None:
        train_data = None
    else:
        full_learning_data_frame = pd.concat(learning_files_list[i].raw_data for i in range(len(learning_files_list)))
        full_learning_data_frame = full_learning_data_frame.sample(frac=1)
        train_data = TrainData(selex_str_len=len(learning_files_list[0].raw_data['DNA_Id'].iloc[0]), selex_files_num=len(learning_files_list))
        train_data.set_one_hot_matrix(dna_data=full_learning_data_frame['DNA_Id'],
                                      primary_selex_sequence=learning_files_list[0].primary_selex_sequence)
        train_data.set_enrichment_matrix(enrichment_data=np.asarray(full_learning_data_frame['cycle_matrix']))
    return train_data


def prediction_data_constructor(prediction_file, model_input_size):
    """This function will construct PredictData object. In the first step it will just __init the
    class and afterwards add a OneHot matrix
    :parameter
     - `prediction_file`: PredictionFile object
     - `model_input_size`: int DeepSELEX input model size needed for the prediction file shape
    :returns
     - `prediction_data`: PredictData object which will be used in the prediction process"""
    if prediction_file is None:
        prediction_data = None
    else:
        prediction_data = PredictData(selex_str_len=model_input_size, predict_str_len=len(prediction_file.raw_data['DNA_Id'].iloc[0]))
        prediction_data.set_one_hot_matrix(dna_data=prediction_file.raw_data['DNA_Id'])
    return prediction_data







