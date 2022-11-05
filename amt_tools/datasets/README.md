## Transcription Datasets
A ```TranscriptionDataset``` implements a dataset wrapper that allows for seamless data sampling in a dataset-agnostic manner.
The wrappers are intended to work with datasets as-shipped.
Currently, the following datasets are supported:
- [MAPS](https://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) (```MAPS.py```)
- [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) V1/V2/V3 (```MAESTRO.py```)
- [GuitarSet](https://guitarset.weebly.com/) (```GuitarSet.py```)
 
Of course, these wrappers can each be extended if one wishes to customize the available ground-truth beyond the default configuration provided.

Several parameters affect where a wrapper looks when working with the data and what pre-computation, if any, to perform. 
- ```save_data``` - rather than reacquiring ground-truth and recalculating features each time a track is sampled, the data can be saved to disk by specifying ```save_data=True```
- ```save_loc``` - ground-truth and features will be saved under the directory specified by ```save_loc```
- ```reset_data``` - any pre-computed features and ground-truth associated with the wrapper under ```save_loc``` will be erased by specifying ```reset_data=True```
- ```store_data``` - all ground-truth and features will be computed or loaded only once and stored in RAM for subsequent access if ```store_data=True```

Data that already exists under ```save_loc``` will only be read in if ```save_data=True```.

See ```common.py``` for more details.

Sampling from a combination of datasets will be supported in the future, but the code (```combo.py```) is currently incomplete and untested.
