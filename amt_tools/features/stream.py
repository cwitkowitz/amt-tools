# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>
# Author: Jonathan Driedger <redacted@redacted>

# My imports
from .. import tools

# Regular imports
from abc import abstractmethod

import sounddevice as sd
import numpy as np
import time as t

#import pyaudio
import threading
import warnings
import datetime


class FeatureStream(object):
    """
    Implements a generic feature streaming wrapper.
    """

    def __init__(self, module, buffer_size=1):
        """
        Initialize parameters for the streaming wrapper.

        Parameters
        ----------
        module : FeatureModule
          Feature extraction method to use for streaming features
        buffer_size : int
          Number of frames to keep track of at a time
        """

        self.module = module

        # Buffer fields
        self.buffer = None
        self.buffer_size = buffer_size

        # Stream tracking fields
        self.start_time = None

    @abstractmethod
    def reset_stream(self):
        """
        Perform any steps to reset the stream.
        """

        # Clear the buffer
        self.buffer = list()

        # Perform any steps to stop streaming
        self.stop_streaming()

    @abstractmethod
    def start_streaming(self):
        """
        Begin streaming the audio.
        """

        # Start tracking time
        self.start_time = self.get_current_time()

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
        Determine if the stream has finished.

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

    def get_num_samples_required(self):
        """
        Determine the number of samples required to extract one frame of features.

        Returns
        ----------
        samples_required : int
          Number of samples
        """

        # Take the maximum amount of samples which will result in one frame
        samples_required = self.module.get_sample_range(1)[-1]

        return samples_required

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
        if self.query_buffer_full():
            # Remove the earliest frame
            self.buffer = self.buffer[-self.buffer_size + 1:]

        # Add the new frame to the buffer
        self.buffer += [frame]

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

    def prime_buffer(self, amount):
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

    def query_buffer_full(self):
        """
        Determine if the buffer is full.

        Returns
        ----------
        buffer_full : bool
          Flag indicating the buffer is full
        """

        # Check if the number of buffered frames meets or exceeds the specified size
        buffer_full = len(self.buffer) >= self.buffer_size

        return buffer_full

    def get_buffered_frames(self):
        """
        Retrieve the currently buffered frames.

        Returns
        ----------
        features : dict
            Dictionary containing a frame of features and the corresponding time
        """

        # Collapse the buffered frames into a single representation
        features = np.concatenate(self.buffer, axis=-1)

        # Get the current time of the stream
        time = np.array([self.get_elapsed_time()])

        # Package the features into a dictionary
        features = tools.dict_unsqueeze({tools.KEY_FEATS : features,
                                         tools.KEY_TIMES : time})

        return features

    @staticmethod
    def get_current_time(decimals=3):
        """
        Determine the current system time.

        Parameters
        ----------
        decimals : int
          Number of digits to keep when rounding

        Returns
        ----------
        current_time : float
          Current system time
        """

        # Get the current time and round to the specified number of digits
        current_time = round(t.time(), decimals)

        return current_time

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
            elapsed_time = self.get_current_time(decimals) - self.start_time

            # Round to the specified number of digits
            elapsed_time = round(elapsed_time, decimals)

        return elapsed_time


class MicrophoneStream(FeatureStream, threading.Thread):
    """
    Implements a streaming wrapper which interfaces with a microphone.
    """
    def __init__(self, module, continuity_successive_frames=True, buffer_size_hardware=1024, issue_warnings=False):
        """
        Initialize parameters for the microphone streaming interface.

        Parameters
        ----------
        See FeatureStream class for others...
        continuity_successive_frames : TODO
          TODO - not sure
        buffer_size_hardware : TODO
          TODO - not sure
        issue_warnings : TODO
          TODO - not sure
        """

        FeatureStream.__init__(self, module, continuity_successive_frames)

        threading.Thread.__init__(self)
        self.setDaemon(daemonic=True)

        self.issue_warnings = issue_warnings

        self.bufferSizeHardware = bufferSizeHardware
        self.stream = None
        self.paudio = None

        self.bufferLen = 4 * frameLen
        self.buffer = np.zeros(self.bufferLen)
        self.startIdxFrameInBuffer = self.bufferLen

        self.running = True

        self.openStreamFromMic()
        self.start()

    def find_input_device(self):
        """
        TODO

        Returns
        ----------
        device_index : TODO
          TODO - not sure
        """

        deviceIndex = None
        for i in range(self.paudio.get_device_count()):
            devinfo = self.paudio.get_device_info_by_index(i)
            print("Device %d: %s" % (i, devinfo["name"]))

            for keyword in ["mic", "input"]:
                if keyword in devinfo["name"].lower():
                    print("Found an input: device %d - %s" % (i, devinfo["name"]))
                    deviceIndex = i
                    return deviceIndex

        if deviceIndex is None:
            print("No preferred input found; using default input device.")

        return deviceIndex

    def open_stream_from_mic(self):
        """
        TODO
        """

        self.paudio = pyaudio.PyAudio()
        deviceIndex = self.findInputDevice()

        self.stream = self.paudio.open(format=pyaudio.paInt16,
                                       channels=1,
                                       rate=self.fs,
                                       input=True,
                                       input_device_index=deviceIndex,
                                       frames_per_buffer=self.bufferSizeHardware)

    def extract_frame_features(self):
        """
        TODO

        Returns
        ----------
        frame : TODO
          TODO - not sure
        """

        if self.continuitySuccessiveFrames:
            while self.startIdxFrameInBuffer > self.bufferLen - self.frameLen:
                continue
            currStrtIdx = self.startIdxFrameInBuffer
            self.startIdxFrameInBuffer = self.startIdxFrameInBuffer + self.hop
            return self.buffer[currStrtIdx: currStrtIdx + self.frameLen]
        else:
            self.startIdxFrameInBuffer = 0
            return self.buffer[:self.frameLen]

    def run(self):
        """
        TODO
        """

        while self.running:
            numOfAvailableSamples = self.stream.get_read_available()
            # assert numOfAvailableSamples <= self.bufferLen
            if numOfAvailableSamples == 0:
                continue
            chunkData = self.stream.read(numOfAvailableSamples)
            chunkAudio = np.fromstring(chunkData, dtype=np.int16)
            # sample is a signed short in +/- 32768.
            # normalize it to +/- 1.0
            chunkAudio = chunkAudio / RANGE_SHORT_VAL

            self.buffer = np.roll(self.buffer, -numOfAvailableSamples)
            self.buffer[-numOfAvailableSamples:] = chunkAudio
            self.startIdxFrameInBuffer = self.startIdxFrameInBuffer - numOfAvailableSamples

            if self.continuitySuccessiveFrames:
                if self.startIdxFrameInBuffer < 0:
                    self.startIdxFrameInBuffer = self.bufferLen
                    if self.issue_warnings:
                        warnings.warn("Buffer was not properly consumed in time.")
                        currTime = time.time()
                        ts = datetime.datetime.fromtimestamp(currTime).strftime('%Y-%m-%d %H:%M:%S')
                        print("Warning: Buffer was not properly consumed in time: " + str(ts))

    def stop(self):
        """
        TODO
        """

        self.running = False

    def close_stream(self):
        """
        TODO
        """

        self.stream.stop_stream()
        self.stream.close()
        self.paudio.terminate()

    def query_finished(self):
        """
        TODO

        Returns
        ----------
        finished : TODO
          TODO - not sure
        """

        return not self.running


class AudioStream(FeatureStream):
    """
    Implements a streaming wrapper which processes audio in real-time.
    """
    def __init__(self, module, buffer_size=1, audio=None, real_time=False, playback=False):
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

        FeatureStream.__init__(self, module, buffer_size)

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

        # Clear the feature buffer
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

        if self.playback and self.real_time and self.audio is not None:
            # Play the audio
            sd.play(self.audio, self.module.sample_rate)

    def stop_streaming(self):
        """
        Stop streaming the audio.
        """

        # Stop tracking time
        super().stop_streaming()

        if self.playback and self.real_time:
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
            sample_time = (self.current_sample + self.get_num_samples_required()) / self.module.sample_rate

            if self.real_time:
                # Wait until it is time to acquire the next frame
                # TODO - add warning when model is not fast enough for real-time
                while self.get_elapsed_time() < sample_time:
                    continue

            # Slice the audio at the boundaries
            audio = self.audio[..., self.current_sample : self.current_sample + self.get_num_samples_required()]

            # Perform feature extraction
            features = self.module.process_audio(audio)

            # Advance the current sample pointer
            self.current_sample += self.module.get_hop_length()

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
    def __init__(self, module, buffer_size=1, audio_path=None, real_time=False, playback=False):
        """
        Initialize parameters for the audio file streaming interface.

        Parameters
        ----------
        See AudioStream class for others...
        audio_path : string
          Path to audio to stream
        """

        # Load the audio at the specified path, with no normalization by default
        audio, _ = tools.load_normalize_audio(audio_path, fs=module.sample_rate, norm=None)

        # Call the parent class constructor - the rest of the functionality is the same
        AudioStream.__init__(self, module, buffer_size, audio, real_time, playback)
