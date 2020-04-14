import numpy as np
import pandas as pd


def predict_prediction_file(model, data, cmd_args):
    """This function will enter the model the prediction_file sequences and the saves the ouput with
    the location which was gave at the cmd
    :parameter
     - `model`: Sequential trained DL CNN model
     - `data`: PredictData object
     - `cmd_args`: The cmd_command line collected by argparse.
    """

    if cmd_args.output_file_location:
        prediction = prediction_loop(model, data)
        max_last_4 = np.amax(prediction[-1], 1)
        max_last_3 = np.amax(prediction[-2], 1)
        min_first_0 = np.amin(prediction[0], 1)
        df = pd.DataFrame()

        df['Results'] = max_last_4 + max_last_3 - min_first_0
        try:
            df.to_csv(cmd_args.output_file_location, index=False, header=False)
        except FileNotFoundError:
            raise FileNotFoundError('The supplied address: {cmd_args.results_location}'
                                    ' to save the results doesnt exist on your machine')

    else:
        print('The user did not supplied results location thus the program will end before calculating the results\n'
              'please use -ofl switch')
        return


def prediction_loop(model, data):
    """This function predicts the results for the sequences from the prediction_file
    the first step is initializing "matrix" which actually a list of 2d matrices. after that each place
    in the list is the results of a different label from the prediction process.

    :parameter
     - `model`: Sequential trained DL CNN model
     - `data`: PredictData object

    :returns
     - `matrix`: A list of matrices
    """

    files_num = model.layers[-1].output_shape[1]
    ht_str_len = model.layers[0].input_shape[1]
    matrix = []
    selex_files_num = model.layers[-1].output_shape[1]
    for i in range(0,files_num):
        matrix.append(np.asarray([[0.5 for i in range(0, data.one_hot_data.shape[0])] for j in range(data.num_of_str)]).T)


    for i in range(0, data.num_of_str):
        predictions = model.predict(data.one_hot_data[:, data.selex_str_len * i:data.selex_str_len * (i + 1), :])
        for index in range(0, selex_files_num):
            matrix[index][:, i] = predictions[:, index]

    return matrix