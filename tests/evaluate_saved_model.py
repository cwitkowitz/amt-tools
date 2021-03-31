# My imports
from amt_models.datasets import MAESTRO_V1, MAPS
from amt_models.features import MelSpec

from amt_models import validate
from amt_models.transcribe import *
from amt_models.evaluate import *

import amt_models.tools as tools

# Regular imports
import torch
import os

model_path = os.path.join(tools.DEFAULT_EXPERIMENTS_DIR,
                          'OnsetsFrames2_MAESTRO_V1_MelSpec',
                          'models', 'model-840.pt')
model = torch.load(model_path)
model.change_device(0)

# Initialize the default piano profile
profile = tools.PianoProfile()

# Processing parameters
sample_rate = 16000
hop_length = 512
dim_in = 229

# Create the mel spectrogram data processing module
data_proc = MelSpec(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_mels=dim_in,
                    htk=True)

# Create a dataset corresponding to the MAESTRO testing partition
mstro_test = MAESTRO_V1(splits=['test'],
                        hop_length=hop_length,
                        sample_rate=sample_rate,
                        data_proc=data_proc,
                        profile=profile,
                        store_data=False)

# Create a dataset corresponding to the MAPS testing partition
# Need to reset due to HTK Mel-Spectrogram spacing
maps_test = MAPS(splits=['ENSTDkAm', 'ENSTDkCl'],
                 hop_length=hop_length,
                 sample_rate=sample_rate,
                 data_proc=data_proc,
                 profile=profile,
                 store_data=False,
                 reset_data=False)

# Initialize the estimation pipeline
validation_estimator = ComboEstimator([NoteTranscriber(profile=profile),
                                       PitchListWrapper(profile=profile)])

# Initialize the evaluation pipeline
evaluators = [LossWrapper(),
              MultipitchEvaluator(),
              NoteEvaluator(key=tools.KEY_NOTE_ON),
              NoteEvaluator(offset_ratio=0.2, key=tools.KEY_NOTE_OFF)]
validation_evaluator = ComboEvaluator(evaluators)

# Get the average results for the MAESTRO testing partition
results = validate(model, mstro_test, evaluator=validation_evaluator, estimator=validation_estimator)

# Print the average results
print('MAESTRO Results')
print(results)

# Reset the evaluator
validation_evaluator.reset_results()

# Get the average results for the MAPS testing partition
results = validate(model, maps_test, evaluator=validation_evaluator, estimator=validation_estimator)

# Print the average results
print('MAPS Results')
print(results)
