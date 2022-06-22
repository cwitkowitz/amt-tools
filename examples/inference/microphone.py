# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.features import MicrophoneStream, WaveformWrapper, MelSpec

import amt_tools.tools as tools

# Feature extraction parameters
sample_rate = 16000
hop_length = 2048

# Initialize the feature extraction protocol
audio_wrp = WaveformWrapper(sample_rate=sample_rate,
                            hop_length=hop_length,
                            center=False)
data_proc = MelSpec(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_mels=229,
                    decibels=True,
                    center=False)

# Disable toolbar globally
tools.global_toolbar_disable()
# Create a figure for the waveform to continually update
wav_visualizer = tools.WaveformVisualizer(figsize=(15, 5),
                                          sample_rate=sample_rate,
                                          time_window=1.0)
# Create a figure for the TFR to continually update
tfr_visualizer = tools.TFRVisualizer(figsize=(15, 5),
                                     sample_rate=sample_rate,
                                     hop_length=hop_length,
                                     time_window=1.0,
                                     plot_frequency=3,
                                     n_bins=229)

# Instantiate the audio stream and start streaming
feature_stream = MicrophoneStream(audio_wrp, audio_buffer_size=5*sample_rate)
# Start the feature stream
feature_stream.start_streaming()

print('Press ENTER to stop...', end=' ')

while not feature_stream.query_finished():
    # Advance the buffer and get the current audio
    samples = feature_stream.extract_frame_features().squeeze(-1)
    # Update the waveform visualizer with the new samples
    wav_visualizer.update(samples)
    # Compute the Mel spectrogram of the audio
    feats = data_proc.process_audio(samples)[0]
    # Update the TFR visualizer with the new frame
    tfr_visualizer.update(feats)
