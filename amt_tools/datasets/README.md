## Transcription Datasets
A ```TranscriptionDataset``` implements a wrapper for a dataset to allow for seamless data sampling.
The wrappers are intended to work with whatever state or structure a dataset is distributed in.
Currently, the following datasets are supported:
 - [MAPS](https://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) (```MAPS.py```)
 - [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) V1/V2/V3 (```MAESTRO.py```)
 - [GuitarSet](https://guitarset.weebly.com/) (```GuitarSet.py```)
 
Rather than reacquiring ground-truth and recalculating features each time a track is sampled, this data can be saved to disk by specifying ```save_data=True```.

The ground truth and features will be saved under the directory specified by ```save_loc```.
As long as the data exists under ```save_loc```, it will be read in.
However, if ```store_data=True``` all ground_truth and features will be stored in RAM for quick access.

By specifying ```reset_data=True```, e.g. if the feature extraction protocol changes, saved data will be overwritten with freshly computed features and ground truth.

See ```common.py``` for more details.

Sampling from a combination of datasets will be supported in the future, but the code is currently incomplete and untested.
