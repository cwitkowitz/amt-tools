## Configurable AMT Pipeline

### Training
The training loop structure is defined in ```train.py```.
At fixed intervals (checkpoints), the model weights and optimizer state are saved, and any validation steps are performed.
Training can be stopped and resumed at a saved model checkpoint.
Any losses or performance scores computed during training and validation are logged with [tensorboard](https://www.tensorflow.org/tensorboard).

See ```train.py``` for more details.

### Transcription (Estimation)
Post-processing, e.g., converting frame-level multi-pitch activations to note predictions, is implemented within a more general estimation framework.
Post-processing is invoked during validation and evaluation, and can be customized based on the specifics of a problem.
At a high-level, an ```Estimator``` transforms the final output of a model into some other data representation and defines steps for writing predictions to disk.
Most ```Estimator``` modules can be parameterized to some degree.
In the example provided, a ```NoteTranscriber``` would transform multi-pitch and potentially onset activations produced by a model into note predictions.

In situtations where more than one type of post-processing is to be performed, ```Estimator``` modules can be chained sequentially using a ```ComboEstimator```.
For instance, a ```MultiPitchRefiner``` can be placed after a ```NoteTranscriber``` in order to overwrite esimated multi-pitch activations with the multi-pitch activations derived from note predictions.

See ```transcribe.py``` and ```inference.py``` for more details.

### Evaluation
The procedure for validation and evaluation is identical and fully customizable.
An ```Evaluator``` wraps an evaluation procedure for a specific prediction type, e.g. notes, such that the results for all tracks in the validation or evaluation set can be stored and averaged.
Some wrappers utilize [mir_eval](https://craffel.github.io/mir_eval/) to perform the main evaluation steps, whereas others perform evaluation steps explicitly.

In situations where more than one type of evaluation is to take place, ```Evaluator``` modules can be combined using a ```ComboEvaluator```.
For instance, a ```NoteEvaluator``` with ```offset_ratio = None``` and a ```NoteEvaluator``` with ```offset_ratio = 0.2``` can be combined to compute ```Note``` and ```Note w/ Offset``` transcription scores, respectively.

See ```evaluate.py``` for more details.
