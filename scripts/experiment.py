# My imports
from train_classifier import train_classifier
from transcribe_classifier import transcribe_classifier
from evaluate import evaluate, write_and_print_results
from utils import *

# Regular imports
from sacred import Experiment

import numpy as np
import os

# TODO - multithreading?
ex = Experiment('6-Fold Cross Validation')

@ex.config
def config():
    win_len = 512 # samples

    # Number of samples between frames
    hop_len = 512 # samples

    # GPU to use for convolutional sparse coding
    gpu_num = 0

    iters = 3500

    batch_size = 300

    l_rate = 1e0

    # Minimum number of frames a positive frame prediction must span to become a note
    min_note_span = 5

    # Switch for console printing in addition to saving text to file
    verbose = False

    seed = 0

@ex.automain
def run_all(win_len, hop_len, gpu_num, iters, batch_size, l_rate, min_note_span, verbose, seed):

    reset_generated_dir(GEN_CLASS_DIR, [], True)

    metrics = []
    for k in range(6):
        player = '0' + str(k)
        class_dir = 'excl_' + player

        train_classifier('', player, win_len, hop_len, gpu_num, iters, batch_size, l_rate, seed)
        transcribe_classifier('', player, class_dir, win_len, hop_len, gpu_num, seed, min_note_span)
        fold_metrics = evaluate('', player, hop_len, verbose)

        metrics += [fold_metrics]

    # TODO - cleanup using .values()
    # Obtain the average value across tracks for each metric
    f_pr = np.mean([np.mean(metrics[i]['f_pr']) for i in range(6)])
    f_re = np.mean([np.mean(metrics[i]['f_re']) for i in range(6)])
    f_f1 = np.mean([np.mean(metrics[i]['f_f1']) for i in range(6)])
    n1_pr = np.mean([np.mean(metrics[i]['n1_pr']) for i in range(6)])
    n1_re = np.mean([np.mean(metrics[i]['n1_re']) for i in range(6)])
    n1_f1 = np.mean([np.mean(metrics[i]['n1_f1']) for i in range(6)])
    n2_pr = np.mean([np.mean(metrics[i]['n2_pr']) for i in range(6)])
    n2_re = np.mean([np.mean(metrics[i]['n2_re']) for i in range(6)])
    n2_f1 = np.mean([np.mean(metrics[i]['n2_f1']) for i in range(6)])

    # Construct a path for the overall results
    ovr_results_path = os.path.join(GEN_RESULTS_DIR, f'overall.txt')

    # Open the file with writing permissions
    ovr_results_file = open(ovr_results_path, 'w')

    # Add heading to file
    write_and_print(ovr_results_file, 'Overall Results\n', verbose)

    # Print the average results across this run
    write_and_print_results(ovr_results_file, verbose, f_pr, f_re, f_f1,
                            n1_pr, n1_re, n1_f1, n2_pr, n2_re, n2_f1)

    # Close the results file
    ovr_results_file.close()