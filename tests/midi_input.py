import amt_models.tools as tools

import numpy as np

batched_notes = tools.load_notes_midi(tools.HOME + '/Downloads/test.mid')
batched_notes_old = tools.load_notes_midi_old(tools.HOME + '/Downloads/test.mid')

pitches, intervals = tools.batched_notes_to_notes(batched_notes_old)

times = np.arange(1000) * 512 / 44100

profile = tools.PianoProfile()

multi_pitch = tools.notes_to_multi_pitch(pitches, intervals, times, profile)

tools.visualize_multi_pitch(multi_pitch)
