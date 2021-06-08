## Tools
This subpackage contains various utilities used by the rest of the framework.

### Instrument Profiles
An ```InstrumentProfile``` defines any parameters necessary to represent data for a particular instrument.
This is helpful to have when performing operations on data at several different stages which are abstracted from each other.
It allows us to avoid making assumptions, such as the pitch mapping, when interpreting the meaning of various data representations.

For instance, a ```GuitarProfile``` can be utilized to define the open-string ```tuning```, as well as the pitch range for each string (```num_frets```).
This information is necessary to convert between different data formats.

See ```instrument.py``` for more details.

### I/O
Currently, the following input operations are supported:
 - Load and normalize audio
 - Read notes from a ```.jams``` file
 - Read pitch activity from a ```.jams``` file
 - Read notes from a ```.midi``` file

When reading from ```.jams``` files, the notes or pitch activity from different sources (e.g. strings) can optionally be combined into a single collection.

Currently, the following output operations are supported:
 - Write pitch predictions to a ```.txt``` file
 - Write note predictions to a ```.txt``` file
 - Write text to a ```.txt``` file and the console (optional)

Some functions are also written for streaming remote files and file management.

See ```io.py``` for more details.

### Data Formatting
There are several data formats used by the framework.
Conversion between most valid pairs is supported.

The following formats exist for representing ```N``` notes:
 - ```notes```

    Tuple containing ```(pitches, intervals)```, where ```pitches``` is a size ```(N)``` array containing the pitches, and ```intervals``` is a size ```(N x 2)``` array containing the onset-offset pairs.

 - ```batched_notes```

    Size ```(N x 3)``` array containing note onset-offset-pitch by row.

The following formats exist for representing pitch activity (including onsets and offsets) across ```T``` frames:
 - ```multipitch```

    Size ```(F x T)``` salience map for ```F``` discrete frequencies.

 - ```pitch_list```

    Size ```(T)``` list of variable-size arrays containing active frequencies.

 - ```tablature```

    Size ```(S x T)``` array representing class membership for ```S``` sources (e.g. strings).

The pitch activity data formats assume there is a corresponding array ```times``` of size ```(T)``` which contains the start time of each frame.

Most data formats can also be placed within ```stacked``` representations, where the same-format data for multiple sources (e.g. strings) is held in a single ```dict```.

In representations where frequency is explicitly defined, either MIDI or Hertz can be used, and there are functions to convert between them.

See ```utils.py``` for more details.

### Data Manipulation
Several functions exist to manipulate data in the following ways:
 - Perform RMS-normalization on audio
 - Sort or slice a note collection given some criteria
 - Blur, normalize, threshold, or framify a salience representation
 - Perform inhibition or remove small blips within a salience representation

See ```utils.py``` for more details.

### Utility
The rest of the utility functions help make the framework more neat and readable. Many are concerned with handling ```dict``` objects, which are used heavily across the framework

See ```utils.py``` for more details.

### Constants
Constants used across the framework are defined in ```constants.py```.

### Visualization
Visualization tools are currently incomplete and largely unused for now.

See ```visualize.py``` for more details.
