# My imports
import amt_models.tools as tools

# Regular imports
from tensorboardX import SummaryWriter
from tqdm import tqdm

import numpy as np
import torch
import os


def file_sort(file_name):
    """
    Augment file names for sorting within the models directory. Since, e.g.,
    /'500/' will by default be scored as higher than /'1500/'. One way to fix
    this is by adding the length of the file to the beginning of the string.

    Parameters
    ----------
    file_name : str
      Path being sorted

    Returns
    ----------
    sort_name : str
      Character count concatenated with original file name
    """

    # Takes into account the number of digits by adding string length
    sort_name = str(len(file_name)) + file_name

    return sort_name


def validate(model, dataset, evaluator, estimator=None):
    """
    Implements the validation or evaluation loop for a model and dataset partition.
    Optionally save predictions and log results.

    Parameters
    ----------
    model : TranscriptionModel
      Model to validate or evalaute
    dataset : TranscriptionDataset
      Dataset (partition) to use for validation or evaluation
    estimator : Estimator
      TODO
    evaluator : Evaluator
      TODO

    Returns
    ----------
    results : dict
      Dictionary containing all relevant results averaged across all tracks
    """

    # Make sure the model is in evaluation mode
    model.eval()

    # Turn off gradient computation
    with torch.no_grad():
        # Loop through the validation track ids
        for track_id in dataset.tracks:
            # Obtain the track data
            track = dataset.get_track_data(track_id)
            if estimator is not None:
                # Transcribe the track
                predictions = estimator.estimate(model, track)
            else:
                batch = track_to_batch(track)
                predictions = model.run_on_batch(batch)
                predictions = track_to_cpu(predictions)
            # Evaluate the predictions
            track_results = evaluator.evaluate(predictions, track)
            # Add the results to the dictionary
            results = append_results(results, track_results)

    # Average the results from all tracks
    results = average_results(results)

    return results


def train(model, train_loader, optimizer, iterations, checkpoints=0, log_dir='.', scheduler=None,
          resume=True, single_batch=False, val_set=None, estimator=None, evaluator=None, vis_fnc=None):
    """
    Implements the training loop for an experiment.

    Parameters
    ----------
    model : TranscriptionModel
      Model to train
    train_loader : DataLoader
      PyTorch Dataloader object for retrieving batches of data
    optimizer : Optimizer
      PyTorch Optimizer for updating weights - expected to have only one parameter group
    iterations : int
      Number of loops through the dataset;
      Each loop may be comprised of multiple batches;
      Each loop contains a snippet of each track exactly once
    checkpoints : int
      Number of equally spaced save/validation checkpoints - 0 to disable
    log_dir : str
      Path to directory for saving model, optimizer state, and events
    scheduler : Scheduler or None (optional)
      PyTorch Scheduler used to update learning rate
    resume : bool
      Start from most recently saved model and optimizer state
    single_batch : bool
      Only process the first batch within each validation loop
    val_set : TranscriptionDataset or None (optional)
      Dataset to use for validation loops
    estimator : Estimator
      TODO
    evaluator : Evaluator
      TODO
    patterns : None or list of str
      String patterns for logging - only results with names containing these patterns will be logged
      If None, all results are logged
    vis_fnc : function(model, i)
      Function to perform any visualization steps during validation loop

    Returns
    ----------
    model : TranscriptionModel
      Trained model
    """

    # TODO - multi_gpu - DataParallel removes ability to call .run_on_batch()
    #                  - Can DataParallel be hooked into TranscriptionModel to
    #                    only call the forward() function?

    # Initialize a writer to log any reported results
    writer = SummaryWriter(log_dir)

    # Start at iteration 0 by default
    start_iter = 0

    if resume:
        # Obtain the files that already exist in the log directory
        log_files = os.listdir(log_dir)

        # Extract and sort files pertaining to the model
        model_files = sorted([path for path in log_files if 'model' in path], key=file_sort)
        # Extract and sort files pertaining to the optimizer state
        optimizer_files = sorted([path for path in log_files if 'opt-state' in path], key=file_sort)

        # Check if any previous checkpoints exist
        if len(model_files) > 0 and len(optimizer_files) > 0:
            # Get the path to the latest model file
            model_path = os.path.join(log_dir, model_files[-1])
            # Get the path to the latest optimizer state file
            optimizer_path = os.path.join(log_dir, optimizer_files[-1])

            # Make the start iteration the iteration when the checkpoint was taken
            start_iter = int(''.join([ch for ch in model_files[-1] if ch.isdigit()]))
            # Get the iteration for the latest optimizer state
            optimizer_iter = int(''.join([ch for ch in optimizer_files[-1] if ch.isdigit()]))
            # Make sure these iterations match
            assert start_iter == optimizer_iter

            # Load the latest model and replace the parameterized version
            model = torch.load(model_path)
            # Replace the randomly initialized parameters with the saved parameters
            # TODO - allow for saving/loading of optimizer with multiple parameter groups
            super(type(optimizer), optimizer).__init__(model.parameters(), optimizer.defaults)
            # Load the latest optimizer state into the parameterized version
            optimizer.load_state_dict(torch.load(optimizer_path))

    # Make sure the model is in training mode
    model.train()

    for global_iter in tqdm(range(start_iter, iterations)):
        # List of losses for each batch in the loop
        train_loss = []
        # Loop through the dataset
        for batch in train_loader:
            # Zero the accumulated gradients
            optimizer.zero_grad()
            # Get the predictions and loss for the batch
            preds = model.run_on_batch(batch)
            # Add the loss to the list of batch losses
            batch_loss = preds['loss']
            train_loss.append(batch_loss.item())
            # Compute gradients
            batch_loss.backward()
            # Perform an optimization step
            optimizer.step()

            if scheduler is not None:
                # Perform a learning rate scheduler step
                scheduler.step()

            # Run additional steps required by model
            model.special_steps()

            if single_batch:
                # Move onto the next iteration after the first batch
                break

        # Average the loss from all of the batches within this loop
        train_loss = np.mean(train_loss)
        # Log the loss
        writer.add_scalar(f'train/loss', train_loss, global_step=global_iter+1)

        # Local iteration of this training sequence
        local_iter = global_iter - start_iter

        # Set a boolean representing whether current iteration is a checkpoint
        if checkpoints == 0:
            checkpoint = False
        else:
            checkpoint = (local_iter + 1) % (iterations // checkpoints) == 0

        # Boolean representing whether training has been completed
        done_training = (global_iter + 1) == iterations

        # If we are at a checkpoint, or we have finished training
        if checkpoint or done_training:
            # Save the model
            torch.save(model, os.path.join(log_dir, f'model-{global_iter + 1}.pt'))
            # Save the optimizer sate
            torch.save(optimizer.state_dict(), os.path.join(log_dir, f'opt-state-{global_iter + 1}.pt'))

            # If visualization protocol was specified, follow it
            if vis_fnc is not None:
                vis_fnc(model, global_iter + 1)

            # If we are at a checkpoint, and a validation set with an estimator is available
            if checkpoint and val_set is not None and evaluator is not None:
                # Validate the current model weights
                validate(model, val_set, evaluator, estimator)
                # Make sure the model is back in training mode
                model.train()

    return model