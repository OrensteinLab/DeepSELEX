import numpy as np
import pandas as pd
import sys


class File:
    """
    A class used to read a file

    Attributes
    ----------
    address : str
        the desired file address
    name : str
        the desired file name if it was given in a full address

    """
    def __init__(self, file_address):
        self.address = file_address
        self.name = file_address.split('/')[-1]


class LearningFile(File):
    """
    A class used to read a learning file (HT-SELEX) for the model, inherits from File

    Attributes
    ----------
    address : str
        the desired file address
    name : str
        the desired file name if it was given in a full address
    num_lines : int
        the number of lines of each selex file
    raw_data : DataFrame
        the DataFrame as was readed from the given SELEX files, it will be added the appropriate cycle label

    Methods
    -------
    read_data()
        read the given files and store it as a data frame
    cycle_matrix(file_serial_number: int, total_learning_files: int)
        add to raw_data DataFrame the correct enrichment marix
    """
    def __init__(self, file_address, primary_selex_sequence):
        super().__init__(file_address)
        self.primary_selex_sequence = primary_selex_sequence
        self.num_lines = 10000
        self.raw_data = self.read_data()

    def read_data(self):
        """Read the file as csv file

        Raises
        ------
        FileNotFoundError
            if the given address is incorrect
        """
        try:
            return pd.read_csv(self.address, sep="\t",
                        header=None,
                        nrows=self.num_lines,
                        names=['DNA_Id'])
        except FileNotFoundError:
            raise FileNotFoundError(f'check if the address: {self.address} contains the desired file')

    def cycle_matrix(self, file_serial_number, total_learning_files):
        """Add to raw_data DataFrame the correct enrichment marix.
            i.e. if the file is a cycle_0 file the matrix will be such as:
            [1,0,0,0,0]
            .
            .
            .
            [1,0,0,0,0]

        Parameters
        ----------
        file_serial_number : int
            The enrichment matrix will be '1' in this location
        file_serial_number : int
            The enrichment will be in total_learning_files width

        """
        arr = [0] * total_learning_files
        arr[file_serial_number] = 1
        cycle_matrix = [np.array(arr) for y in range(self.num_lines)]
        self.raw_data['cycle_matrix'] = cycle_matrix


class PredictionFile(File):
    """
    A class used to read a prediction file (PBM, ChIP-seq..) for the model, inherits from File

    Attributes
    ----------
    address : str
        the desired file address
    name : str
        the desired file name if it was given in a full address
    raw_data : DataFrame
        the DataFrame as was readed from the given prediction file

    Methods
    -------
    read_data()
        read the given files and store it as a data frame
    """
    def __init__(self, file_address):
        super().__init__(file_address)
        self.raw_data = self.read_data()

    def read_data(self):
        """Read the file as csv file

        Raises
        ------
        FileNotFoundError
            if the given address is incorrect
        """
        try:
            return pd.read_csv(self.address, sep="\t",
                        header=None,
                        names=['DNA_Id'])
        except FileNotFoundError:
            raise FileNotFoundError(f'check if the address: {self.address} contains the desired file')


class LinkerFile(File):
    """
    A class used to read a learning file (HT-SELEX) for the model, inherits from File
    It is called from the create_data module
    Attributes
    ----------
    address : str
        the desired file address
    name : str
        the desired file name if it was given in a full address
    raw_data : DataFrame
        the DataFrame as was readed from the given SELEX files, it will be added the appropriate cycle label
    primary_selex_sequence : str
        the sequence which is the HT-SELEX experiment primary sequence.
        If the selex file is of the form: Alx4_TGGTAG20NCG_P_0.fastq, the primary sequence is: TGGTAG20NCG
    Methods
    -------
    read_data()
        read the given files and store it as a data frame
    get_linker_sequnces()
        Returns the linker sequences as reported at the HT-SELEX original report
    """
    def __init__(self, file_address, primary_selex_sequence):
        super().__init__(file_address)
        self.raw_data = self.read_data()
        self.primary_selex_sequence = primary_selex_sequence

    def read_data(self):
        """Read the file as csv fileReturns the linker sequences as reported at the HT-SELEX original article.

        Raises
        ------
        FileNotFoundError
            if the given address is incorrect
        """
        try:
            return pd.read_excel(self.address)
        except FileNotFoundError:
            raise FileNotFoundError('Make sure the linker selex file is in the project folder')

    def get_linker_sequnces(self):
        """Returns the linker sequences as reported at the HT-SELEX original article.
        Raises
        ------
        IndexError
            if the given primary_selex_sequence is incorrect and dosent exist in the excel
        """
        if self.primary_selex_sequence:
            try:
                full_string = self.raw_data[self.raw_data['Name'] == self.primary_selex_sequence].iloc[0][
                'Sequence']
                start_linker = full_string[:full_string.index('N')]
                end_linker = full_string[full_string.rfind('N') + 1:]
            except IndexError:
                raise IndexError(f'The primary_selex_sequence: {self.primary_selex_sequence} as was inserted at the command line as -pss doesnt exist in the linker excel')
        else:
            start_linker = end_linker = None
        return start_linker, end_linker

def selex_linker_sequence(file_address, primary_selex_sequence):
    """read the linker HT-SELEX excel and stores it in selex_linker file object
        Afterwards the linker sequences are extracted via the LinkerFile object

    :parameter
      - `file_address`: The linker file address.
      - `primary_selex_sequence: the sequence which is the HT-SELEX experiment primary sequence.
        If the selex file is of the form: Alx4_TGGTAG20NCG_P_0.fastq, the primary sequence is: TGGTAG20NCG
        this sequence should be supplied in the cmd
    :returns
     - Returns both the start_linker and end_linker which are concatenated to the HT-SELEX sequences"""
    selex_linker_file = LinkerFile(file_address, primary_selex_sequence)
    return selex_linker_file.get_linker_sequnces()


def files_constructor(cmd_args, file_type):
    """read both learning and prediction files via
    panda read_csv.
    :parameter
      - `cmd_args`: The cmd_command line collected by argparse.
      - `file_type: If learning the function will return the learning file list,
                    if it prediction the function will return the predictio file.
                    Third option is allowed to be added by the user.
    :returns
     - the function returns either a learning files list or a prediction file"""

    if file_type == 'learning':
        learning_files_list = [LearningFile(address, cmd_args.primary_selex_sequence) for address in cmd_args.learning_file_list]
        [learning_files_list[i].cycle_matrix(i, len(learning_files_list)) for i in range(len(learning_files_list))]
        return learning_files_list

    elif file_type == 'prediction':
        if cmd_args.prediction_file:
            return PredictionFile(cmd_args.prediction_file)
        else:
            return None

    else:
        'the user can insert here some code for suppeltementary files'


def model_files(cmd_args):
    """This function serves as the constructor of the data's objects.
    In the end of the process two object will be returned. learing_data
    and prediction_data.
    :parameter
      - `cmd_args`: The cmd_command line collected by argparse.

    :returns
     - `learning_files_list`: A list of LearningFile objects
     - `prediction_file`: PredictionFile object"""
    if cmd_args.learning_file_list:
        learning_files_list = files_constructor(cmd_args, file_type='learning')
    else:
        learning_files_list = None

    if cmd_args.prediction_file:
        prediction_file = files_constructor(cmd_args, file_type='prediction')
    else:
        prediction_file = None


    return learning_files_list, prediction_file






