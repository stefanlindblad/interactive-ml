from helper.load_data import read_input_data, read_truth_data, read_input_file, read_truth_file
from helper.save_data import write_output_file, write_output_array, write_output_array2
import os


def convert_folder(foldername):
    files = os.listdir(foldername)
    for entry in files:
        filename = foldername + entry
        if "groundtruth" in entry:
            T = read_truth_file(filename)
            write_output_array(filename, T)
        if "input" in entry:
            R,G,B,D = read_input_file(filename)
            write_output_array2(filename, R, G, B, D)

def load_models(modellocation):
    models = []
    with open(modellocation) as f:
    for line in f:
        if line.endswith("\n"):
            line = line[:-1]
        models.append(line)
    return models

############################

modellocation = "models.txt"
datalocation = "D:/train_data/"

modellist = load_models(modellocation)
for model in modellist:
    foldername = datalocation + model + "/"
    convert_folder(foldername)
