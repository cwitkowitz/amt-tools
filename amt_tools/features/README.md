## Feature Extraction
A ```FeatureModule``` implements a wrapper for a feature extraction protocol, abstracting the choice of features from the rest of the pipeline.
Wrappers are provided for common feature extraction methods, including:
- ```CQT``` - Constant-Q Transform (CQT)
- ```HCQT``` - Harmonic CQT (HCQT)
- ```VQT``` - Variable-Q Transform (VQT)
- ```HVQT``` - Harmonic VQT (HVQT)
- ```STFT``` - Short-Time Fourier Transform (STFT)
- ```MelSpec``` - Mel-Scaled Spectrogram
- ```WaveformWrapper``` - Raw Waveform Frames
- ```SignalPower``` - Frame-Level Signal Power

An instantiated feature extraction module can be reused for multiple tracks, and clear definitions exist regarding, e.g., how many frames will be generated from audio with a specific number of samples.
Each wrapper utilizes [librosa](https://librosa.org/doc/latest/index.html) to perform the main feature extraction steps.

See ```common.py``` for more details.

Concatenation or stacking of features produced from multiple feature extraction modules will be supported in the future, but the code (```combo.py```) is currently incomplete and may produce unexpected behavior.

## Feature Streaming
A ```FeatureStream``` can be used to compute features using the above modules in a real-time or online fashion.
The following feature streaming protocols are available in ```stream.py```:
- ```MicrophoneStream``` - process frames of microphone audio in real-time
- ```AudioStream``` - process pre-existing audio in an online fashion and mimic real-time processing
- ```AudioFileStream``` - open and process audio file in an online fashion and mimic real-time processing
