# DeepSELEX
Inference of DNA-binding preferences from HT-SELEX data using deep neural networks
The flags for the command line interface:
    :parameter

      - `learning_file_list`: A list of HT-SELEX files. Should be written as follows:
        -lfl Alx4_TGGTAG20NCG_P_0.fastq Alx4_TGGTAG20NCG_P_1.fastq Alx4_TGGTAG20NCG_P_2.fastq Alx4_TGGTAG20NCG_P_3.fastq Alx4_TGGTAG20NCG_P_4.fastq

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
Command line example:
python main.py -lfl Alx4_TGGTAG20NCG_P_0.fastq Alx4_TGGTAG20NCG_P_1.fastq Alx4_TGGTAG20NCG_P_2.fastq Alx4_TGGTAG20NCG_P_3.fastq Alx4_TGGTAG20NCG_P_4.fastq
-pss TGGTAG20NCG -pf Alx4_1744.1_deBruijn.txt -ofl output_file_name.csv -sml saved_model_name.h5
