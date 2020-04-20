# DeepSELEX
Inference of DNA-binding preferences from HT-SELEX data using deep neural networks

The flags for the command line interface:

      - `learning_file_list`: A list of HT-SELEX files. Should be written as follows:
        -lfl demo_data/ALX4_TGTGTC20NGA_W_0.fastq demo_data/ALX4_TGTGTC20NGA_W_1.fastq demo_data/ALX4_TGTGTC20NGA_W_2.fastq demo_data/ALX4_TGTGTC20NGA_W_3.fastq demo_data/ALX4_TGTGTC20NGA_W_4.fastq

      - `primary_selex_sequence: the sequence which is the HT-SELEX experiment primary sequence.
        If the selex file is of the form: ALX4_TGTGTC20NGA_W_0.fastq, the primary sequence is: TGTGTC20NGA
        this sequence should be supplied in the cmd.  Should be written as follows:
        -pss TGTGTC20NGA

      - `pf: Prediction data file.
        Should be written as follows:
        -pf demo_data/Alx4_1744.1_deBruijn.txt or any other predicted file

      - `ofl: The output file name and location.
        Should be written as follows:
        -ofl output_file_name.csv

      - `sml: If supply, saves the model in the supplied address.
        Should be written as follows:
        -sml saved_model_name.h5

      - `lml: Loads the model from the supplied address
        Should be written as follows:
        -lml loaded_model_name.h5
Training command line example:
python deep_selex.py -lfl demo_data/ALX4_TGTGTC20NGA_W_0.fastq demo_data/ALX4_TGTGTC20NGA_W_1.fastq demo_data/ALX4_TGTGTC20NGA_W_2.fastq demo_data/ALX4_TGTGTC20NGA_W_3.fastq demo_data/ALX4_TGTGTC20NGA_W_4.fastq -pss TGTGTC20NGA -pf demo_data/Alx4_1744.1_deBruijn.txt -ofl output_file_name.csv -sml saved_model_name.h5
Using pre-trained model command line example:
python deep_selex.py -lml test_model.h5 -pf demo_data/Alx4_1744.1_deBruijn.txt -ofl delete_this.csv
