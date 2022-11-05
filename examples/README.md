## Examples
The ```papers``` subdirectory contains standalone scripts which implement the following papers:
- [Onsets & Frames](https://arxiv.org/abs/1710.11153) (```of_1.py```)
- [Onsets & Frames 2](https://arxiv.org/abs/1810.12247) (```of_2.py```)
- [TabCNN](https://archives.ismir.net/ismir2019/paper/000033.pdf) (```tabcnn.py```)

Each script instantiates, trains, and evaluates the respective model on the respective dataset(s).

The ```inference``` subdirectory so far contains only one script, ```microphone.py```, which demonstrates how one might interact with and visualize audio in real-time using ```amt-tools``` (see ```features```/```tools``` subpackages for more details).

## Usage
Each example script can be run from the command line as follows:
```
python <path_to_script>/<script>.py
```

Mutable parameters are defined at the top of each example script.

## Generated Files (Paper Scripts)
Execution of a script will generate the following under ```<root_dir>```:
- ```n/``` - folder (beginning at ```n = 1```)<sup>1</sup> containing [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) experiment files:
  - ```config.json``` - parameter values used for the experiment
  - ```cout.txt``` - contains any text printed to console
  - ```metrics.json``` - evaluation results for the experiment
  - ```run.json``` system and experiment information
- ```models/``` - folder containing saved model and optimizer state at each checkpoint, as well as an events file (for each execution) readable by [tensorboard](https://www.tensorflow.org/tensorboard)
- ```estimated/``` - folder containing final predictions for each track within the test set
- ```results/``` - folder containing separate evaluation results for each track within the test set
- ```_sources/``` - folder containing copies of scripts at the time(s) execution

<sup>1</sup>An additional folder (```n += 1```) containing similar files is created for each execution with the same experiment name ```<EX_NAME>```.

## Analysis (Paper Scripts)
During training, losses and various validation metrics can be analyzed in real-time by running:
```
tensorboard --logdir=<root_dir>/models --port=<port>
```
Here we assume the current working directory contains ```<root_dir>```, and ```<port>``` is an integer corresponding to an available port (```port = 6006``` if unspecified).

After running the above command, navigate to [http://localhost:&lt;port&gt;]() with an internet browser to view any reported training or validation observations within the tensorboard interface.
