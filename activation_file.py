import os
import sys
sys.path.insert(1, '../git')
import main


file_0 = 'C:/Users/asifm/Desktop/sd/script_folder/round_predict/HT-SELEX/Alx4_TGGTAG20NCG_P_0.fastq'
file_1 = 'C:/Users/asifm/Desktop/sd/script_folder/round_predict/HT-SELEX/Alx4_TGGTAG20NCG_P_1.fastq'
file_2 = 'C:/Users/asifm/Desktop/sd/script_folder/round_predict/HT-SELEX/Alx4_TGGTAG20NCG_P_2.fastq'
file_3 = 'C:/Users/asifm/Desktop/sd/script_folder/round_predict/HT-SELEX/Alx4_TGGTAG20NCG_P_3.fastq'
file_4 = 'C:/Users/asifm/Desktop/sd/script_folder/round_predict/HT-SELEX/Alx4_TGGTAG20NCG_P_4.fastq'

file_pred = 'C:/Users/asifm/Desktop/sd/script_folder/round_predict/PBM/Cell08/Alx4_1744.1_deBruijn'
x=1
os.system(f'python C:/Users/asifm/Desktop/sd/script_folder/round_predict/git/main.py -lf {file_0} {file_1} {file_2} {file_3} {file_4} -p {file_pred}')
