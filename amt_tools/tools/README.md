## Tools
The ```tools``` subpackage contains various utilities used by the rest of the framework, the most important of which are described below.

### Data Representations
The following formats can be used to represent ```K``` notes:
- ```notes``` - tuple containing ```(pitches, intervals)```, where ```pitches``` is a size ```(K)``` array containing pitches, and ```intervals``` is a size ```(K x 2)``` array containing onset-offset pairs
- ```batched_notes``` - size ```(K x 3)``` array containing note onsets, offsets, and pitch by row

The following formats can be used to represent discrete pitch activity (including onset and offset activity) across ```N``` frames:
- ```multi_pitch``` - size ```(F x N)``` array of activations for ```F``` discrete frequencies
- ```tablature``` - size ```(S x N)``` array of class membership for ```S``` degrees of freedom (e.g. strings)
- ```logistic``` - size ```(C x N)``` array of activations for ```C``` distinct note sources (e.g. string/fret pairs)

The following formats can be used to represent continuous-valued pitch activity across ```N``` frames:
- ```pitch_list``` - length ```(N)``` list of variable-size arrays containing continuous-valued frequencies

When working with data representations for pitch activity, it is typically assumed there is a corresponding array ```times``` of size ```(N)``` which contains the time associated with each frame.

Most data can also be placed within a ```stacked``` representation, where same-format data for multiple sources (e.g. strings) is stored in a single collection, i.e. a dictionary or array.

Many functions are available to manipulate the data of the various representations or to convert between them.

See ```utils.py``` for more details.

### I/O
Currently, the following input operations are supported:
- Load and (optionally) normalize audio
- Read notes from a ```.jams``` or ```.midi``` file
- Read pitch activity from a ```.jams``` file

When reading from ```.jams``` files, the notes or pitch activity corresponding to different sources (e.g. strings) can optionally be encoded within a ```stacked``` representation.

Currently, the following output operations are supported:
- Write pitch activity to a ```.txt``` file
- Write notes to a ```.txt``` file
- Write list contents to a ```.txt``` file
- Write text to a ```.txt``` file and (optionally) the console

Some functions are also available for file management and streaming remote files.

See ```io.py``` for more details.

### Instrument Profiles
An ```InstrumentProfile``` defines the attributes necessary to represent data associated with a particular instrument, e.g. the lowest and highest playable pitch.
Profiles are helpful to have when operating on data at several different levels of abstraction across the framework.
They circumvent the need to make assumptions, such as the pitch mapping for an array of multi-pitch activations, when interpreting the meaning of various data representations.

For instance, a ```GuitarProfile``` can be instantiated with an open-string ```tuning``` and ```num_frets```, implicitly defining the pitch range of the instrument as well as each string.
This profile can then be referenced across the framework when constructing and interpreting data representations.

See ```instrument.py``` for more details.

### Visualization
Static visualization functions are implemented for most data representations.
Additionally, a ```Visualizer``` can be employed to iteratively update plots as new data becomes available during online processing.

See ```visualize.py``` for more details.

### Constants
Constants used across the framework are defined in ```constants.py```.
