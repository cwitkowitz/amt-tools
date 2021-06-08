## Feature Extraction
This subpackage contains wrappers for common feature extraction methods, including:
 - Constant-Q Transform (CQT)
 - Harmonic CQT (HCQT)
 - Variable-Q Transform (VQT)
 - Harmonic VQT (HVQT)
 - Short-Time Fourier Transform (STFT)
 - Mel-Spectrogram

When used in this way, the instantiated feature extraction protocol can be reused for multiple tracks, and clear definitions exist regarding, e.g., how many frames will be generated from audio with a specific number of samples.


See ```common.py``` for more details.

Concatenation or stacking of features produced from multiple protocols will be fully supported in the future, but the code is currently incomplete and may produce unexpected behavior.
