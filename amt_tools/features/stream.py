# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>
# Author: Jonathan Driedger <redacted@redacted>

# My imports
from .. import tools

# Regular imports
from abc import abstractmethod

import sounddevice as sd
import numpy as np

#import pyaudio
import threading
import warnings
import datetime


from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib.pyplot as plt


class FeatureStream(object):
    """
    Implements a generic feature streaming wrapper.
    """

    def __init__(self, module, frame_buffer_size=1):
        """
        Initialize parameters for the streaming wrapper.

        Parameters
        ----------
        module : FeatureModule
          Feature extraction method to use for streaming features
        frame_buffer_size : int
          Number of frames to keep track of at a time
        """

        self.module = module

        # Buffer fields
        self.frame_buffer = None
        self.frame_buffer_size = frame_buffer_size

        # Stream tracking fields
        self.start_time = None

    @abstractmethod
    def reset_stream(self):
        """
        Perform any steps to reset the stream.
        """

        # Clear the buffer
        self.frame_buffer = list()

        # Perform any steps to stop streaming
        self.stop_streaming()

    @abstractmethod
    def start_streaming(self):
        """
        Begin streaming the audio.
        """

        # Start tracking time
        self.start_time = tools.get_current_time()

    @abstractmethod
    def stop_streaming(self):
        """
        Stop streaming the audio.
        """

        # Stop tracking time
        self.start_time = None

    @abstractmethod
    def extract_frame_features(self):
        """
        Acquire the next frame from the stream.
        """

        return NotImplementedError

    @abstractmethod
    def query_active(self):
        """
        Determine if the stream has finished. (Default behavior)

        Returns
        ----------
        active : bool
          Flag indicating the stream is up and running
        """

        # Check if the timer has been started
        active = self.start_time is not None

        return active

    @abstractmethod
    def query_finished(self):
        """
        Determine if the stream has finished.
        """

        return NotImplementedError

    def buffer_new_frame(self, frame=None):
        """
        Add a new frame to the buffer. This function is only required if a buffer
        is needed. Otherwise, extract_frame_features() will be sufficient by itself.

        Parameters
        ----------
        frame : ndarray (Optional)
          Existing frame of features to add to the buffer

        Returns
        ----------
        features : dict
            Dictionary containing a frame of features and the corresponding time
        """

        if frame is None:
            # Get a new frame of features
            frame = self.extract_frame_features()

        # Check if the buffer has exceeded capacity
        if self.query_frame_buffer_full():
            # Remove the earliest frame
            self.frame_buffer = self.frame_buffer[-self.frame_buffer_size + 1:]

        # Add the new frame to the buffer
        self.frame_buffer += [frame]

        # Hand over the updated features
        features = self.get_buffered_frames()

        return features

    def buffer_empty_frame(self):
        """
        Prime the buffer with an empty frame.

        Returns
        ----------
        features : dict
            Dictionary containing a frame of features and the corresponding time
        """

        # Construct a frame filled with zeros
        empty_frame = np.zeros((self.module.get_num_channels(),
                                self.module.get_feature_size(),
                                1)).astype(tools.FLOAT32)

        # Obtain the updated features
        features = self.buffer_new_frame(empty_frame)

        return features

    def prime_frame_buffer(self, amount):
        """
        Add a specified amount of empty frames to the buffer.

        Parameters
        ----------
        amount : int
          Number of empty frames to add
        """

        # Prime the buffer with features
        for i in range(amount):
            # Add empty frames (no need to extract features here)
            self.buffer_empty_frame()

    def query_frame_buffer_full(self):
        """
        Determine if the buffer is full.

        Returns
        ----------
        frame_buffer_full : bool
          Flag indicating the buffer is full
        """

        # Check if the number of buffered frames meets or exceeds the specified size
        frame_buffer_full = len(self.frame_buffer) >= self.frame_buffer_size

        return frame_buffer_full

    def get_buffered_frames(self):
        """
        Retrieve the currently buffered frames.

        Returns
        ----------
        features : dict
            Dictionary containing a frame of features and the corresponding time
        """

        # Collapse the buffered frames into a single representation
        features = np.concatenate(self.frame_buffer, axis=-1)

        # Get the current time of the stream
        time = np.array([self.get_elapsed_time()])

        # Package the features into a dictionary
        features = tools.dict_unsqueeze({tools.KEY_FEATS : features,
                                         tools.KEY_TIMES : time})

        return features

    def get_elapsed_time(self, decimals=3):
        """
        Determine the amount of time elapsed since the start of streaming.

        Parameters
        ----------
        decimals : int
          Number of digits to keep when rounding

        Returns
        ----------
        elapsed_time : float
          Amount of time passed since stream started
        """

        # Default the elapsed time
        elapsed_time = 0

        if self.start_time is not None:
            # Compute the difference between the current time and start time
            elapsed_time = tools.get_current_time(decimals) - self.start_time

            # Round to the specified number of digits
            elapsed_time = round(elapsed_time, decimals)

        return elapsed_time


class MicrophoneStream(FeatureStream, threading.Thread):
    """
    Implements a streaming wrapper which interfaces with a microphone.
    """
    def __init__(self, module, frame_buffer_size=1, sample_buffer_size=None, device=None, enforce_continuity=True):
        """
        Initialize parameters for the microphone streaming interface.

        Parameters
        ----------
        See FeatureStream class for others...
        device : TODO
          TODO - not sure
        sample_buffer_size : TODO
          TODO - not sure
        enforce_continuity : TODO
          TODO - not sure
        """

        FeatureStream.__init__(self, module, frame_buffer_size)
        threading.Thread.__init__(self)
        self.setDaemon(daemonic=True)

        self.stream = None

        if sample_buffer_size is None:
            sample_buffer_size = 10 * self.module.get_num_samples_required()
        self.sample_buffer = np.zeros(sample_buffer_size)

        # TODO parameters
        self.device = None
        self.select_device(device)
        self.enforce_continuity = enforce_continuity

        self.previous_time = None
        self.current_sample = None

        # TODO
        self.reset_stream()
        self.start()

    @staticmethod
    def query_devices(verbose=True):
        devices = sd.query_devices()

        if not verbose:
            devices = [idx for idx in range(len(devices))]

        return devices

    def select_device(self, idx=None):
        available_devices = self.query_devices(False)

        if idx is None or idx in available_devices:
            self.device = idx

    def get_current_device(self):
        if self.device is not None:
            device = sd.query_devices(self.device)
        else:
            device = sd.query_devices(kind='input')

        return device

    def reset_stream(self):
        # Clear the feature buffer and stop streaming
        super().reset_stream()

        self.sample_buffer[:] = 0
        self.previous_time = None
        self.current_sample = len(self.sample_buffer)

        self.stream = sd.InputStream(samplerate=self.module.sample_rate,
                                     blocksize=None,
                                     device=self.device,
                                     channels=1,
                                     dtype=None)

    def start_streaming(self):
        if self.query_finished():
            self.reset_stream()

        # Start tracking time
        super().start_streaming()

        if self.stream is not None:
            self.stream.start()

    def stop_streaming(self, pause=False):
        self.previous_time = self.get_elapsed_time()

        # Stop tracking time
        super().stop_streaming()

        if self.stream is not None:
            self.stream.stop()
            if not pause:
                self.stream.close()

    def run(self):
        while self.query_active():
            print('In Thread')
            num_samples_available = self.stream.read_available
            # assert numOfAvailableSamples <= self.bufferLen

            if num_samples_available > 0:
                # TODO - abstract normalization type
                new_audio = tools.rms_norm(self.stream.read(num_samples_available)[0][:, 0])

                self.sample_buffer = np.roll(self.sample_buffer, -num_samples_available)
                self.sample_buffer[-num_samples_available:] = new_audio

                if self.enforce_continuity:
                    self.current_sample = min(0, self.current_sample - num_samples_available)

                    # TODO - add warning when model is not fast enough for real-time (current_sample <= 0)

    def extract_frame_features(self):
        """
        #fig = plt.figure()
        #ax = SubplotZero(fig, 111)
        #fig.add_subplot(ax)
        #plt.show()
        while True:
            print(self.sample_buffer[:])
            #ax.cla()
            #ax.plot(self.sample_buffer)
        print('Not supposed to be here')
        """
        if self.enforce_continuity:
            while self.current_sample > len(self.sample_buffer) - self.module.get_num_samples_required():
                continue
            audio = self.sample_buffer[self.current_sample : self.current_sample + self.module.get_num_samples_required()]
            self.current_sample += self.module.get_hop_length()
        else:
            audio = self.sample_buffer[-self.module.get_num_samples_required():]

        # Perform feature extraction
        features = self.module.process_audio(audio)

        #features = np.zeros((self.module.get_num_channels(), self.module.get_feature_size(), 1)).astype(tools.FLOAT32)

        return features

    def query_active(self):
        active = self.stream.active

        return active

    def query_finished(self):
        finished = self.stream.closed

        return finished

    def get_elapsed_time(self, decimals=3):
        elapsed_time = super().get_elapsed_time()

        if self.previous_time is not None:
            elapsed_time += self.previous_time

        elapsed_time = round(elapsed_time, decimals)

        return elapsed_time


class AudioStream(FeatureStream):
    """
    Implements a streaming wrapper which processes audio in real-time.
    """
    def __init__(self, module, frame_buffer_size=1, audio=None, real_time=False, playback=False):
        """
        Initialize parameters for the audio streaming interface.

        Parameters
        ----------
        See FeatureStream class for others...
        audio : ndarray
          Mono-channel audio to stream
        real_time : bool
          Whether to process the audio in a real-time fashion
        playback : bool
          Whether to playback the audio when streaming in real-time
        """

        FeatureStream.__init__(self, module, frame_buffer_size)

        # Audio streaming parameters
        self.audio = None
        self.current_sample = None

        # Real-time parameters
        self.playback = playback
        self.real_time = real_time

        # Reset audio and streaming markers
        self.reset_stream(audio)

    def reset_stream(self, audio=None):
        """
        Initialize parameters for the audio streaming interface.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio to stream (replaces existing audio)
        """

        # Clear the feature buffer and stop streaming
        super().reset_stream()

        # Reset the current sample
        self.current_sample = 0

        if audio is not None:
            # Replace the current audio
            self.audio = audio

    def start_streaming(self):
        """
        Begin streaming the audio.
        """

        # Start tracking time
        super().start_streaming()

        if self.playback and self.audio is not None:
            # Play the audio
            sd.play(self.audio, self.module.sample_rate)

    def stop_streaming(self):
        """
        Stop streaming the audio.
        """

        # Stop tracking time
        super().stop_streaming()

        if self.playback:
            # Stop playing audio
            sd.stop(ignore_errors=True)

    def extract_frame_features(self):
        """
        Acquire the next frame of features from the stream.

        Returns
        ----------
        features : ndarray
          Features for one frame of audio
        """

        # Default the features
        features = None

        # Check if the stream is active and if there are more features to acquire
        if self.query_active() and not self.query_finished():
            # Determine the nominal time of the last sample needed to extract the frame
            sample_time = (self.current_sample + self.module.get_num_samples_required()) / self.module.sample_rate

            if self.real_time:
                # Wait until it is time to acquire the next frame
                # TODO - add warning when model is not fast enough for real-time
                while self.get_elapsed_time() < sample_time:
                    continue

            # Slice the audio at the boundaries
            audio = self.audio[..., self.current_sample : self.current_sample + self.module.get_num_samples_required()]

            # Advance the current sample pointer
            self.current_sample += self.module.get_hop_length()

            # Perform feature extraction
            features = self.module.process_audio(audio)

        return features

    def query_finished(self):
        """
        Determine if the stream has finished.

        Returns
        ----------
        finished : bool
          Flag indicating there are no frames left to process
        """

        # Default finished to true
        finished = True

        if self.audio is not None:
            # Determine if the counter has exceeded the number of available samples
            finished = self.current_sample > len(self.audio)

        return finished


class AudioFileStream(AudioStream):
    """
    Implements a streaming wrapper which processes an audio file in real-time.
    """
    def __init__(self, module, frame_buffer_size=1, audio_path=None, real_time=False, playback=False):
        """
        Initialize parameters for the audio file streaming interface.

        Parameters
        ----------
        See AudioStream class for others...
        audio_path : string
          Path to audio to stream
        """

        # Load the audio at the specified path, with no normalization by default
        # TODO - abstract normalization type
        audio, _ = tools.load_normalize_audio(audio_path, fs=module.sample_rate, norm=None)

        # Call the parent class constructor - the rest of the functionality is the same
        AudioStream.__init__(self, module, frame_buffer_size, audio, real_time, playback)
