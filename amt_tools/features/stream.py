# Author: Jonathan Driedger <redacted@redacted>
# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .. import tools

# Regular imports
from abc import abstractmethod

import sounddevice as sd
import numpy as np
import time as t

import pyaudio
import threading
import warnings
import datetime


class FeatureStream(object):
    """
    Implements a generic feature streaming wrapper.
    """

    def __init__(self, module, continuity_successive_frames=True):
        """
        Initialize parameters for the streaming wrapper.

        Parameters
        ----------
        module : FeatureModule
          Feature extraction method to use for streaming features
        continuity_successive_frames : TODO
          TODO - not sure
        TODO - context_length for buffering groups of frames
        """

        self.module = module
        self.continuity_successive_frames = continuity_successive_frames

    @abstractmethod
    def extract_frame_features(self):
        """
        Acquire the next frame from the stream.
        """

        return NotImplementedError

    @abstractmethod
    def query_finished(self):
        """
        Determine if the stream has finished.
        """

        return NotImplementedError

    def get_num_samples_required(self):
        """
        Determine the number of samples required to extract one frame of features.
        """

        # Take the maximum amount of samples which will result in one frame
        samples_required = self.module.get_sample_range(1)[-1]

        return samples_required


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
    def __init__(self, module, audio=None, real_time=False, playback=False):
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

        FeatureStream.__init__(self, module, True)

        # Audio streaming parameters
        self.audio = None
        self.start_time = None
        self.current_sample = None

        # Reset audio and streaming markers
        self.reset_audio(audio)

        # Real-time parameters
        self.playback = playback
        self.real_time = real_time

    def reset_audio(self, audio=None):
        """
        Initialize parameters for the audio streaming interface.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio to stream
        """

        if audio is not None:
            # Replace the current audio
            self.audio = audio

        # Reset the current sample and start time
        self.start_time = None
        self.current_sample = 0

    def start_streaming(self):
        """
        Begin streaming the audio.
        """

        if self.playbackAudio and self.realTime and self.audio is not None:
            # Play the audio
            sd.play(self.audio, self.module.sample_rate)

        # Start tracking time
        self.start_time = self.get_current_time()

    def stop_streaming(self):
        """
        Stop streaming the audio.
        """

        if self.playbackAudio and self.realTime:
            # Stop playing audio
            sd.stop()

        # Stop tracking time
        self.start_time = None

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

        # Check if there are more features to acquire
        if not self.query_finished():
            # Determine the nominal time of the current sample pointer
            sample_time = self.current_sample / self.module.sample_rate

            if self.real_time:
                # Wait until it is time to acquire the next frame
                while self.get_current_time() < sample_time:
                    continue

            # Determine the audio boundaries for the next frame
            start_frame = self.current_sample
            self.current_sample += self.get_num_samples_required()

            # Slice the audio using the boundaries
            audio = self.audio[..., start_frame : self.current_sample]

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

        if self.audio is not None:
            # Determine if the counter has exceeded the number of available samples
            # TODO - plus half buffer size / 2
            finished = self.current_sample > len(self.audio)

        return finished

    def get_time_passed(self, decimals=3):
        """
        Determine the amount of time elapsed since the start of streaming.

        Parameters
        ----------
        decimals : int
          Number of digits to keep when rounding

        Returns
        ----------
        time_passed : float
          Time elapsed
        """

        # Default the elapsed time
        time_passed = 0

        if self.start_time is not None:
            # Compute the difference between the current time and start time
            time_passed = self.get_current_time(decimals) - self.start_time

            # Round to the specified number of digits
            time_passed = round(time_passed, decimals)

        return time_passed

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


class AudioFileStream(AudioStream):
    """
    Implements a streaming wrapper which processes an audio file in real-time.
    """
    def __init__(self, module, audio_path=None, real_time=False, playback=False):
        """
        Initialize parameters for the audio file streaming interface.

        Parameters
        ----------
        See AudioStream class for others...
        audio_path : string
          Path to audio to stream
        """

        # Load the audio at the specified path
        audio, _ = tools.load_normalize_audio(audio_path, fs=module.sample_rate)

        # Call the parent class constructor - the rest of the functionality is the same
        AudioStream.__init__(self, module, audio, real_time, playback)
