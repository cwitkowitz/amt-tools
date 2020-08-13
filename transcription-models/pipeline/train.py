# My imports
from pipeline.transcribe import *
from pipeline.evaluate import *

from tools.utils import *

# Regular imports
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import os


def file_sort(file_name):
    """
    Augment file names for sorting within the models directory. Since, e.g.,
    /'500/' will by default be scored as higher than /'1500/', we need to fix
    this by adding the length of the file to the beginning of the string.

    Parameters
    ----------
    file_name: str
      Path being sorted

    Returns
    ----------
    sort_name : str
      Character count concatenated with original file name
    """

    # Takes into account the decimal place by adding string length
    sort_name = str(len(file_name)) + file_name
    return sort_name


def train(model, train_loader, optimizer, iterations,
          checkpoints=0, log_dir='.', val_set=None,
          scheduler=None, resume=False, single_batch=False):
    # TODO - multi-gpu
    # TODO - scheduler

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

            # TODO - seed random state as well?
            model = torch.load(model_path)
            optimizer.load_state_dict(torch.load(optimizer_path))

    # How many new iterations to run
    new_iters = iterations - start_iter

    for global_iter in tqdm(range(start_iter, iterations)):
        train_loss = []
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model.run_on_batch(batch)
            batch_loss = torch.mean(preds['loss'])
            train_loss.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()

            # TODO - this is only for OF - how to abstract? - put run_special_steps() in TranscriptionModel? ->exactly
            #clip_grad_norm_(model.parameters(), 3)

            if single_batch:
                # Move onto the next iteration after the first batch
                break

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

                model.eval()
                with torch.no_grad():
                    val_results = get_results_format()
                    for track in val_set:
                        track = val_set.slice_track(track)
                        predictions = transcribe(model, track)
                        track_results = evaluate(predictions, track)
                        val_results = add_result_dicts(val_results, track_results)

                    val_results = average_results(val_results)

                    for type in val_results.keys():
                        if isinstance(val_results[type], dict):
                            for metric in val_results[type].keys():
                                writer.add_scalar(f'val/{type}/{metric}', val_results[type][metric], global_step=global_iter+1)
                        else:
                            writer.add_scalar(f'val/{type}', val_results[type], global_step=global_iter)
                model.train()

    return model
