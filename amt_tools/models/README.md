## Transcription Models
A ```TranscriptionModel``` defines a neural network intended for music transcription, as well as any pre-processing or post-processing steps.
Currently, the following models are implemented and available for use or modification:
 - [Onsets & Frames](https://arxiv.org/abs/1710.11153) (```onsetsframes.py```)
 - [Onsets & Frames 2](https://arxiv.org/abs/1810.12247) (```onsetsframes.py```)
 - [TabCNN](https://archives.ismir.net/ismir2019/paper/000033.pdf) (```tabcnn.py```)

The [Onsets & Frames](https://arxiv.org/abs/1710.11153) acoustic model and music language model are defined explicitly, and therefore can be used independently within other architectures.

An ```OutputLayer``` is used to make the final layer of a model interchangeable, such that the backbone model can be trained for different purposes.
Currently, the following output layers are implemented:
 - ```SoftmaxGroups```

    Performs classification across independent monophonic sources (e.g. strings).
    [TabCNN](https://archives.ismir.net/ismir2019/paper/000033.pdf) uses this output layer to predict which fret (including none) is held down during each frame for each string.
 
 - ```LogisticBank```

    Performs binary classification across independent sources which are either active or inactive (e.g. keys).
    [Onsets & Frames](https://arxiv.org/abs/1710.11153) uses this output layer to predict which pitches are active during each frame.

This abstraction is useful, e.g., if we wish to use the [Onsets & Frames](https://arxiv.org/abs/1710.11153) model to predict guitar tablature.
Doing this would be as simple as:
```
onsetsframes.adjoin[-1] = SoftmaxGroups(dim_lm, num_strings, num_frets + 1)
```
This line replaces the ```LogisticBank``` in the refined multipitch prediction head with a ```SoftmaxGroups```.
Here, ```dim_lm``` refers to the embedding size of the output of the preceding language model in the refined multipitch prediction head. 

See ```common.py``` for more details.
