# My imports
from pipeline.transcribe import *
from pipeline.evaluate import *

from tools.utils import *

# Regular imports
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import os


def file_sort(file_name):
    """
    Augment file names for sorting within the models directory. Since, e.g.,
    /'500/' will by default be scored as higher than /'1500/'. One way to fix
    this is by adding the length of the file to the beginning of the string.

    Parameters
    ----------
    file_name: str
      Path being sorted

    Returns
    ----------
    sort_name : str
      Character count concatenated with original file name
    """

    # Takes into account the number of digits by adding string length
    sort_name = str(len(file_name)) + file_name

    return sort_name


def validate(model, dataset, estim_dir=None, results_dir=None):
    """
    Augment file names for sorting within the models directory. Since, e.g.,
    /'500/' will by default be scored as higher than /'1500/'. One way to fix
    this is by adding the length of the file to the beginning of the string.

    Parameters
    ----------
    file_name: str
      Path being sorted

    Returns
    ----------
    sort_name : str
      Character count concatenated with original file name
    """

    # Make sure the model is in evaluation mode
    model.eval()

    # Create a dictionary to hold the results
    results = get_results_format()

    with torch.no_grad():
        # Loop through the validation track ids
        for track_id in dataset.tracks:
            # Obtain the track data
            track = dataset.get_track_data(track_id)
            # Transcribe the track
            predictions = transcribe(model, track, estim_dir)
            # Evaluate the predictions
            track_results = evaluate(predictions, track, results_dir)
            # Add the results to the dictionary
            results = add_result_dicts(results, track_results)

    # Average the results from all tracks
    results = average_results(results)

    return results


def train(model, train_loader, optimizer, iterations,
          checkpoints=0, log_dir='.', val_set=None,
          scheduler=None, resume=True, single_batch=False):
    # TODO - multi_gpu - DataParallel removes ability to call .run_on_batch()

    # Initialize a writer to log training loss and validation loss/results
    writer = SummaryWriter(log_dir)

    start_iter = 0
    if resume:
        log_files = os.listdir(log_dir)
        model_files = sorted([path for path in log_files if 'model' in path], key=file_sort)
        optimizer_files = sorted([path for path in log_files if 'opt-state' in path], key=file_sort)

        if len(model_files) > 0 and len(optimizer_files) > 0:
            model_path = os.path.join(log_dir, model_files[-1])
            optimizer_path = os.path.join(log_dir, optimizer_files[-1])

            start_iter = int(''.join([ch for ch in model_files[-1] if ch.isdigit()]))
            optimizer_iter = int(''.join([ch for ch in optimizer_files[-1] if ch.isdigit()]))

            assert start_iter == optimizer_iter

            model = torch.load(model_path)
            optimizer.load_state_dict(torch.load(optimizer_path))

    # How many new iterations to run
    new_iters = iterations - start_iter

    for global_iter in tqdm(range(start_iter, iterations)):
        train_loss = []
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model.run_on_batch(batch)
            batch_loss = preds['loss']
            train_loss.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            model.special_steps()

            if single_batch:
                # Move onto the next iteration after the first batch
                break

        # Average the loss from all of the batches
        train_loss = np.mean(train_loss)
        writer.add_scalar(f'train/loss', train_loss, global_step=global_iter+1)

        # Local iteration of this training sequence
        local_iter = global_iter - start_iter

        # Set a boolean representing whether current iteration is a checkpoint
        if checkpoints == 0:
            checkpoint = False
        else:
            checkpoint = (local_iter + 1) % (new_iters // checkpoints) == 0

        # Boolean representing whether training has been completed
        done_training = (global_iter + 1) == iterations

        # If we are at a checkpoint, or we have finished training
        if checkpoint or done_training:
            # Save the model
            torch.save(model, os.path.join(log_dir, f'model-{global_iter + 1}.pt'))
            # Save the optimizer sate
            torch.save(optimizer.state_dict(), os.path.join(log_dir, f'opt-state-{global_iter + 1}.pt'))

            if checkpoint and val_set is not None:
                val_results = validate(model, val_set)
                log_results(val_results, writer, global_iter + 1, ['loss', 'f1-score'])

    return model
