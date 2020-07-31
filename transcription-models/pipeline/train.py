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
    # Takes into account the decimal place by adding string length
    return str(len(file_name)) + file_name

def train(classifier, train_loader, optimizer, iterations,
          checkpoints=None, log_dir='.', val_set=None,
          scheduler=None, resume=False, single_batch=False):
    # TODO - multi-gpu
    # TODO - scheduler

    writer = SummaryWriter(log_dir)

    start_iter = 0
    if resume:
        log_files = os.listdir(log_dir)
        classifier_files = sorted([path for path in log_files if 'classifier' in path], key=file_sort)
        optimizer_files = sorted([path for path in log_files if 'opt-state' in path], key=file_sort)

        if len(classifier_files) > 0 and len(optimizer_files) > 0:
            classifier_path = os.path.join(log_dir, classifier_files[-1])
            optimizer_path = os.path.join(log_dir, optimizer_files[-1])

            start_iter = int(''.join([ch for ch in classifier_files[-1] if ch.isdigit()]))
            optimizer_iter = int(''.join([ch for ch in optimizer_files[-1] if ch.isdigit()]))

            assert start_iter == optimizer_iter

            classifier = torch.load(classifier_path)
            optimizer.load_state_dict(torch.load(optimizer_path))

    if checkpoints is None:
        checkpoints = iterations / 10

    for i in tqdm(range(start_iter, iterations)):
        train_loss = []
        for batch in train_loader:
            optimizer.zero_grad()
            _, batch_loss = classifier.run_on_batch(batch)
            batch_loss = torch.mean(batch_loss)
            train_loss.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()

            if single_batch:
                break

        train_loss = np.mean(train_loss)
        writer.add_scalar(f'train/loss', train_loss, global_step=i)

        if ((i + 1) % checkpoints == 0 or i + 1 == iterations) and checkpoints != 0:
            torch.save(classifier, os.path.join(log_dir, f'classifier-{i + 1}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(log_dir, f'opt-state-{i + 1}.pt'))

            if val_set is not None:
                classifier.eval()
                with torch.no_grad():
                    val_results = get_results_format()
                    for track in val_set:
                        # TODO - bring out hop length and min note span
                        predictions = transcribe(classifier, track, hop_length=512, min_note_span=5)
                        track_results = evaluate(predictions, track, hop_length=512)
                        val_results = add_result_dicts(val_results, track_results)

                    val_results = average_results(val_results)
                    val_step = i // checkpoints

                    for type in val_results.keys():
                        if isinstance(val_results[type], dict):
                            for metric in val_results[type].keys():
                                writer.add_scalar(f'val/{type}/{metric}', val_results[type][metric], global_step=val_step)
                        else:
                            writer.add_scalar(f'val/{type}', val_results[type], global_step=val_step)
                classifier.train()

    return classifier
