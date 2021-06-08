## Configurable AMT Pipeline

### Training
The training and validation loop structure is defined in ```train.py```.
At fixed checkpoints, the model weights and optimizer state are saved, and any validation steps are performed.
Training can be stopped and resumed.

See ```train.py``` for more details.

### Transcribing
The transcription process is implemented within a more general estimation framework.
The process is invoked during validation and evaluation, and can be customized based on the specifics of a problem.
At a high-level, an ```Estimator``` transforms the raw output of a model in some way.

An example would be the application of a ```NoteTranscriber``` to transform the raw multipitch salience map produced by a model into note predictions.
Here, ```NoteTranscriber``` performs this transformation using basic heuristics and post-processing.
Most ```Estimator``` objects can be parameterized to some degree.

When more than one procedure is to be performed, ```Estimator``` objects can be stacked sequentially within a ```ComboEstimator```.
For instance, a ```MultiPitchRefiner``` can be placed after a ```NoteTranscriber``` in order to overwrite raw multipitch salience maps with the multipitch activations derived from the note predictions.

See ```transcribe.py``` for more details.

### Evaluating
The validation and evaluation procedure is also fully customizable.
At a high-level, an ```Evaluator``` defines an evaluation procedure for a specific prediction type.

For example, a ```NoteEvaluator``` with ```offset_ratio = 0.2``` can be instantiated to log relevant metrics during validation checkpoints and write final results during evaluation.

When evaluation covers more than one type of prediction, ```Evaluator``` objects can be grouped within a ```ComboEvaluator```.

See ```evaluate.py``` for more details.
