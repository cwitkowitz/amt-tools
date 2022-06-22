# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import FeatureModule

# Regular imports
from librosa.core.constantq import __early_downsample_count as early_downsample_count
from librosa.filters import window_bandwidth, wavelet_lengths

import numpy as np
import librosa


# TODO - the convention for alpha in librosa has changed, potentially invalidating things
#        a lot of things have changed and I should go through and verify this wrapper again

class VQT(FeatureModule):
    """
    Implements a Variable-Q Transform wrapper.
    """
    def __init__(self, sample_rate=22050, hop_length=512, decibels=True,
                 fmin=None, n_bins=84, bins_per_octave=12, gamma=None):
        """
        Initialize parameters for the VQT.

        Parameters
        ----------
        See FeatureModule class for others...
        fmin : float
          Center frequency of lowest filter
        n_bins : int
          Number of frequency bins, starting at fmin
        bins_per_octave : int
          Number of bins per octave
        gamma : float or None
          Bandwidth offset for determining filter lengths
        """

        super().__init__(sample_rate, hop_length, 1, decibels)

        # Default the lowest center frequency to the note C1
        if fmin is None:
            # C1 by default
            fmin = librosa.note_to_hz('C1')
        self.fmin = fmin

        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.window = 'hann'

        # Compute the inverse of what would be the constant Q factor
        self.alpha = 2.0 ** (1.0 / self.bins_per_octave) - 1

        # Default gamma using the procedure defined in
        # librosa.filters.constant_q.vqt documentation
        if gamma is None:
            gamma = 24.7 * self.alpha / 0.108
        self.gamma = gamma

        # Determine the number of octaves does the transform span
        n_octs = int(np.ceil(float(self.n_bins) / self.bins_per_octave))
        self.n_octs = n_octs

    def get_early_ds_count(self):
        """
        Utility function to calculate the number of downsamples required
        before processing the top-octave, given the VQT parameters, based
        on the approach used in librosa.filters.constant_q.vqt.

        TODO - can this be abstracted from librosa.filters.constant_q.vqt?
               the filter cutoff is required as input for early_downsample_count
               I am repeating lines of code already in the function

        Returns
        ----------
        early_ds_count : int
          Number of time we must downsample a signal before applying filters
        """

        # Obtain the highest center frequency for the transform (top-octave)
        fmax = np.max(librosa.cqt_frequencies(n_bins=self.n_bins, fmin=self.fmin,
                                              bins_per_octave=self.bins_per_octave))

        # Calculate the constant-Q factor (assuming gamma=0)
        cQ = 1.0 / (2.0 ** (1. / self.bins_per_octave) - 1)
        # Calculate the constant-Q bandwidth (assuming gamma=0)
        cQ_bandwidth = window_bandwidth(self.window) / cQ
        # Obtain the filter cutoff, this time accounting for gamma
        freq_cutoff = fmax * (1 + 0.5 * cQ_bandwidth) + 0.5 * self.gamma

        # Calculate the Nyquist rate
        nyquist = self.sample_rate / 2.0

        # Calculate the number of downsamples
        early_ds_count = early_downsample_count(nyquist=nyquist,
                                                filter_cutoff=freq_cutoff,
                                                hop_length=self.hop_length,
                                                n_octaves=self.n_octs)

        return early_ds_count

    def get_expected_frames(self, audio):
        """
        Determine the number of frames the module will return
        for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        num_frames : int
          Number of frames expected
        """

        # Calculate the number of downsamples before processing
        early_ds_count = self.get_early_ds_count()

        # Determine total number of downsamples
        k = early_ds_count + self.n_octs - 1
        # Determine the downsampling factors we will use
        k = np.arange(early_ds_count, k + 1)
        # Calculate the signal length associated with these factors
        sig_lens = np.ceil(len(audio) / (2**k))
        # Determine the downsampled hop lengths
        hop_lens = self.hop_length // (2**k)
        # Calculate the number of hops for each downsampling factor
        num_hops = sig_lens // hop_lens
        # Number of frames is the lowest amount of hops plus one
        num_frames = int(min(num_hops + 1))

        return num_frames

    def get_sample_range(self, num_frames):
        """
        Determine the range of audio samples which will produce
        features with a given number of frames.

        Parameters
        ----------
        num_frames : int
          Number of frames for sample-range query

        Returns
        ----------
        sample_range : ndarray
          Valid audio signal lengths to obtain queried number of frames
        """

        # Calculate the number of downsamples before processing
        early_ds_count = self.get_early_ds_count()

        # Downsampling factors
        early_ds_factor = 2**early_ds_count

        # Calculate the boundaries
        max_samples = ((num_frames * self.hop_length // early_ds_factor) - 1) * early_ds_factor
        min_samples = max(1, max_samples - self.hop_length + 1)

        # Construct an array ranging between the minimum and maximum number of samples
        sample_range = np.arange(min_samples, max_samples + 1)

        return sample_range

    def process_audio(self, audio):
        """
        Get the VQT features for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        feats : ndarray
          Post-processed features
        """

        # Calculate the VQT using librosa
        vqt = librosa.vqt(y=audio,
                          sr=self.sample_rate,
                          hop_length=self.hop_length,
                          fmin=self.fmin,
                          n_bins=self.n_bins,
                          bins_per_octave=self.bins_per_octave,
                          gamma=self.gamma)
        # Take the magnitude of the VQT
        vqt = np.abs(vqt)
        # Post-process the VQT
        feats = super().post_proc(vqt)

        return feats

    def get_times(self, audio, at_start=False):
        """
        Determine the time, in seconds, associated with each frame.

        Parameters
        ----------
        audio: ndarray
          Mono-channel audio
        at_start : bool
          Whether time is associated with beginning of frame instead of center

        Returns
        ----------
        times : ndarray
          Time in seconds of each frame
        """

        # Get the times associated with the hops
        times = super().get_times(audio)

        if at_start:
            # Determine the length of the lowest frequency filter
            longest_length = wavelet_lengths(freqs=self.fmin,
                                             sr=self.sample_rate,
                                             window=self.window,
                                             gamma=self.gamma,
                                             alpha=self.alpha)
            # Subtract half of the length of the longest filter
            times -= ((longest_length // 2) / self.sample_rate)

        return times

    def get_feature_size(self):
        """
        Helper function to access dimensionality of features.

        Returns
        ----------
        feature_size : int
          Dimensionality along feature axis
        """

        feature_size = self.n_bins

        return feature_size
