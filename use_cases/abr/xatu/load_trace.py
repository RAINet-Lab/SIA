from pathlib import Path
import os


COOKED_TRACE_FOLDER = str(Path(__file__).resolve().parent / 'train') + '/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_trace_folder = str(cooked_trace_folder)
    if not cooked_trace_folder.endswith('/'):
        cooked_trace_folder += '/'
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names
