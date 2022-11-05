## Transcription Models
A ```TranscriptionModel``` defines a neural network intended for music transcription or related tasks, as well as any pre-processing or post-processing steps.
Currently, the following models are implemented and available for use or modification:
- [Onsets & Frames](https://arxiv.org/abs/1710.11153) (```onsetsframes.py```)
- [Onsets & Frames 2](https://arxiv.org/abs/1810.12247) (```onsetsframes.py```)
- [TabCNN](https://archives.ismir.net/ismir2019/paper/000033.pdf) (```tabcnn.py```)

The acoustic and music language models within the [Onsets & Frames](https://arxiv.org/abs/1710.11153) architecture are defined explicitly, and therefore can be used independently within other architectures.

An ```OutputLayer``` abstracts the final prediction layer of a model, such that it is interchangeable and the backbone model can be trained for different purposes.
Currently, the following output layers are implemented:
- ```SoftmaxGroups``` - performs classification across independent monophonic sources (e.g. strings)
- ```LogisticBank``` - performs binary classification across independent sources which are either active or inactive (e.g. keys)

[TabCNN](https://archives.ismir.net/ismir2019/paper/000033.pdf) uses the ```SoftmaxGroups``` output layer to predict which fret (including none) is held down during each frame of audio for each string.
[Onsets & Frames](https://arxiv.org/abs/1710.11153) uses the ```LogisticBank``` output layer to predict which pitches are active during each frame of audio.
This abstraction is useful, e.g., if we wish to use the [Onsets & Frames](https://arxiv.org/abs/1710.11153) model to predict guitar tablature.
Doing this would be as simple as:
```
onsetsframes.adjoin[-1] = SoftmaxGroups(dim_lm, num_strings, num_frets + 1)
```
This line replaces the ```LogisticBank``` in the refined multi-pitch prediction head with a ```SoftmaxGroups```.
Here, ```dim_lm``` refers to the embedding size of the output of the preceding language model in the refined multi-pitch prediction head. 

See ```common.py``` for more details.
