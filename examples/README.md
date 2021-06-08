## Examples
This subdirectory contains standalone scripts which implement the following papers:
 - [Onsets & Frames](https://arxiv.org/abs/1710.11153) (```of_1.py```)
 - [Onsets & Frames 2](https://arxiv.org/abs/1810.12247) (```of_2.py```)
 - [TabCNN](https://archives.ismir.net/ismir2019/paper/000033.pdf) (```tabcnn.py```)

Each script instantiates, trains, and evaluates the respective model on the respective dataset.

## Usage
Each script can be run from the command line as follows:
```
python <script>.py
```

Parameters for each experiment are defined at the top of the script. 

## Generated Files
The experiment root directory ```<root_dir>``` is one parameter defined at the top of each script.
Execution of a script will generate the following under ```<root_dir>```:
 - ```n/```

    Folder (beginning at ```n = 1```) containing [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) experiment files:
 
     - ```config.json``` - parameter values for the experiment
     - ```cout.txt``` - contains any text printed to console
     - ```metrics.json``` - evaluation results for the experiment
     - ```run.json``` system and experiment information

    An additional folder (```n += 1```) with experiment files is created for each run where the name of the sacred experiment is the same. 

 - ```models/```

    Folder containing saved model and optimizer states at checkpoints, as well as the events file that tensorboard reads.

 - ```estimated/```

    Folder containing frame-level and note-level predictions for all tracks in the test set.
    Predictions are organized within ```.txt``` files according to [MIREX I/O](https://www.music-ir.org/mirex/wiki/2020:Multiple_Fundamental_Frequency_Estimation_%26_Tracking) specifications for transcription.

 - ```results/```

    Folder containing individual evaluation results for each track within the test set.

 - ```_sources/```

    Folder containing copies of the script at the time(s) of execution.

## Analysis
During training, losses and various validation metrics can be analyzed in real-time by running:
```
tensorboard --logdir=<root_dir>/models --port=<port>
```
Here we assume the current directory within the command-line interface contains ```<root_dir>```.
 ```<port>``` is an integer corresponding to an available port (```port = 6006``` if unspecified).

After running the command, navigate to <http://localhost:<port>> to view any reported training or validation observations within the tensorboard interface.
