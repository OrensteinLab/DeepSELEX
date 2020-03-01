import gc

import read_files
import create_data
import build_model
import prediction_module
def read_user_switches_from_cmd():
    """returns the parser.parse_args() which cotains the cmd switches
    The user can add more switches if he desires
    :flags

      - `learning_file_list`: A list of HT-SELEX files. Should be written as follows:
        -lfl demo_data/Alx4_TGGTAG20NCG_P_0.fastq demo_data/Alx4_TGGTAG20NCG_P_1.fastq demo_data/Alx4_TGGTAG20NCG_P_2.fastq Alx4_TGGTAG20NCG_P_3.fastq demo_data/Alx4_TGGTAG20NCG_P_4.fastq

      - `primary_selex_sequence: the sequence which is the HT-SELEX experiment primary sequence.
        If the selex file is of the form: Alx4_TGGTAG20NCG_P_0.fastq, the primary sequence is: TGGTAG20NCG
        this sequence should be supplied in the cmd.  Should be written as follows:
        -pss TGGTAG20NCG

      - `pf: Prediction data file.
        Should be written as follows:
        -pf Alx4_1744.1_deBruijn.txt or any other predicted file

      - `ofl: The output file name and location.
        Should be written as follows:
        -ofl output_file_name.csv

      - `sml: If supply, saves the model in the supplied address.
        Should be written as follows:
        -sml saved_model_name.h5

      - `lml: Loads the model from the supplied address
        Should be written as follows:
        -lml loaded_model_name.h5

    :returns
     - `parser.parse_args()`: A argparse module contains all the switches the user entered
     """
    file_list = ['demo_data/Alx4_TGGTAG20NCG_P_0.fastq', 'demo_data/Alx4_TGGTAG20NCG_P_1.fastq', 'demo_data/Alx4_TGGTAG20NCG_P_2.fastq', 'demo_data/Alx4_TGGTAG20NCG_P_3.fastq', 'demo_data/Alx4_TGGTAG20NCG_P_4.fastq']
    file_pred = 'demo_data/Alx4_1744.1_deBruijn.txt'
    primary_selex_sequence = 'TGGTAG20NCG'
    saved_model_location = 'test_model.h5'
    results_location = 'demo_results.mmm'
    import argparse
    parser = argparse.ArgumentParser(description='DeepSELEX is a Deep-Learning model for learning High Throughtput SELEX data')
    parser.add_argument('-lfl', '--learning_file_list', nargs='+', help='list of learning files, no comma needed', required=False, default=file_list)
    parser.add_argument('-pss', '--primary_selex_sequence', type=str, metavar='', required=False, help='primary SELEX sequence of the selex experiment', default=primary_selex_sequence)
    parser.add_argument('-pf', '--prediction_file', type=str, metavar='', required=False, help='Prediction data file', default=file_pred)
    parser.add_argument('-ofl', '--output_file_location', type=str, metavar='', required=False, help='The output file name and location', default=results_location)
    parser.add_argument('-sml', '--saved_model_location', type=str, metavar='', required=False, help='Saves the model in the supplied address', default=saved_model_location)
    parser.add_argument('-lml', '--loaded_model_location', type=str, metavar='', required=False, help='Loads the model from the supplied address')

    return parser.parse_args()


if __name__ == "__main__":
    """
    This is DeepSELEX main program, welcome!
    The Design pattern of this DeepLearning program is as follows:
    1. Get the cmd switches by the function read_user_switches_from_cmd()
    2. Read the model files and store them in learning_files_list which is a list of LearningFile objects
    and prediction_file which is a PredictionFile object. this is done by read_files.model_files()
    3. Transform the model files into appropriate one hot encode matrix. The learning files will be
    added linker sequences.
    4. Delete the unnecessary files object and use the garbage collector
    5. Train the model or load and existing one
    6. Predict the results (if the user supplied a prediction_file)"""

    cmd_args = read_user_switches_from_cmd()  # 1.

    learning_files_list, prediction_file = read_files.model_files(cmd_args)  # 2.
    train_data, prediction_data = create_data.data_constructor(learning_files_list, prediction_file)  # 3.

    learning_files_list = prediction_file = None  # 4.
    gc.collect()

    model = build_model.manage_model(cmd_args, train_data)  # 5.
    prediction_module.predict_prediction_file(model=model, data=prediction_data, cmd_args=cmd_args)  # 6.
