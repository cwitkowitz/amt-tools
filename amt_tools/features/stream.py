# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>
# Author: Jonathan Driedger

# My imports
from .. import tools

# Regular imports
from abc import abstractmethod

import warnings
import time

try:
    from pynput import keyboard
except ImportError:
    warnings.warn('Could not import keyboard, likely because an X ' +
                  'connection could not be acquired.', category=RuntimeWarning)

try:
    import sounddevice as sd
except OSError:
    warnings.warn('Could not import sounddevice. Please install PortAudio and ' +
                  'try again.\n  >>> sudo apt-get install libportaudio2', category=RuntimeWarning)

import numpy as np
import threading
import librosa

# Tolerance past which we consider ourselves falling behind on processing
MIC_LAG_TOL = 0.250 # seconds


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

        # Perform any steps to stop streaming
        self.stop_streaming()

        # Clear the buffer
        self.frame_buffer = list()

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
            # Index where frames should start being retained
            start_idx = len(self.frame_buffer) - self.frame_buffer_size + 1
            # Remove the earliest frame(s)
            self.frame_buffer = self.frame_buffer[start_idx:]

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
    def __init__(self, module, frame_buffer_size=1, audio_norm=None,
                 audio_buffer_size=None, device=None, enforce_continuity=True,
                 suppress_warnings=True):
        """
        Initialize parameters for the microphone streaming interface.

        Parameters
        ----------
        See FeatureStream class for others...
        audio_norm : float or None
          Type of normalization to perform when loading audio
          -1 - root-mean-square
          See librosa for others...
            - None case is handled here
        audio_buffer_size : int
          Size (in samples) of the audio buffer
        device : int or None (Optional)
          Index of the device to use for data input (see query_devices() for options...)
        enforce_continuity : bool
          TODO - bring this functionality into a separate class (no current_sample, buffer_size = samples_required)
          Whether to extract frames according to explicit hops or by using the most recent samples
        suppress_warnings : bool
          Whether to ignore warning messages
        """

        FeatureStream.__init__(self, module, frame_buffer_size)
        threading.Thread.__init__(self)

        # Kill this thread when the invoking process is complete
        self.setDaemon(daemonic=True)

        # Field for the audio stream
        self.stream = None

        # Type of normalization to perform on each new chunk of audio
        self.audio_norm = audio_norm

        # Create the audio buffer
        if audio_buffer_size is None:
            # Default the audio buffer size
            audio_buffer_size = 4 * self.module.get_num_samples_required()
        self.audio_buffer = np.zeros(audio_buffer_size).astype(tools.FLOAT32)

        # Select the chosen or default device
        self.device = None
        self.select_device(device)

        # Instant vs. continuous streaming
        self.enforce_continuity = enforce_continuity
        # Flag for printing warning messages
        self.suppress_warnings = suppress_warnings

        # Streaming parameters
        self.previous_time = None
        self.current_sample = None

        # Use this flag to indicate when the thread is to be stopped
        self.killed = False

        # Reset buffers and streaming markers
        self.reset_stream()

        # Start the thread
        self.start()

        # Create a new thread to listen for key presses/releases
        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()

    @staticmethod
    def query_devices(verbose=True):
        """
        Obtain a collection of available input and output devices.

        Parameters
        ----------
        verbose : bool
          Flag for printing out device information

        Returns
        ----------
        devices : list of int OR dict of device information
          Available devices as integers or a dictionary of relevant information
        """

        # Query devices using sounddevice library
        devices = sd.query_devices()

        if not verbose:
            # Reduce to a list of indexes
            devices = [idx for idx in range(len(devices))]

        return devices

    def select_device(self, idx=None):
        """
        Choose the device to use when opening the microphone stream.

        Parameters
        ----------
        idx : int or None (Optional)
          Device index - if unspecified, default device is chosen
        """

        # Get a list of possible devices
        available_devices = self.query_devices(False)

        # Make sure the chosen device is valid
        if idx is None or idx in available_devices:
            # Update the corresponding field
            self.device = idx

    def get_current_device(self):
        """
        Obtain information about the device currently chosen.

        Returns
        ----------
        device : dict
          Device information tracked by sounddevice library
        """

        if self.device is None:
            # Default input device
            device = sd.query_devices(kind='input')
        else:
            # Query the current device
            device = sd.query_devices(self.device)

        return device

    def reset_stream(self):
        """
        Clear everything related to any previous streams.
        """

        if self.killed and not self.suppress_warnings:
            # Print a warning message if the thread was already killed, since it cannot be restarted
            warnings.warn('The MicrophoneStream Thread was already killed. A new ' +
                          'MicrophoneStream instance should be created.', category=RuntimeWarning)

        # Stop streaming and clear the feature buffer
        super().reset_stream()

        # Reset streaming parameters
        self.audio_buffer[:] = 0
        self.previous_time = None
        self.current_sample = len(self.audio_buffer)

        # Re-initialize the stream
        self.stream = sd.InputStream(samplerate=self.module.sample_rate,
                                     blocksize=None,
                                     device=self.device,
                                     channels=1,
                                     dtype=tools.FLOAT32)

    def start_streaming(self):
        """
        Begin streaming audio from the microphone.
        """

        if self.killed and not self.suppress_warnings:
            # Print a warning message if the thread was already killed, since it cannot be restarted
            warnings.warn('The MicrophoneStream Thread was already killed. A new ' +
                          'MicrophoneStream instance should be created.', category=RuntimeWarning)

        # Check if the microphone stream was previously closed
        if self.query_finished():
            # If so, reset the stream
            self.reset_stream()

        # Start tracking time
        super().start_streaming()

        if self.stream is not None:
            # If a stream exists, start it
            self.stream.start()

    def stop_streaming(self, pause=False):
        """
        Stop streaming audio from the microphone.

        Parameters
        ----------
        pause : bool
          Whether to pause the stream instead of terminating it
        """

        # Keep track of how much time has passed so far
        self.previous_time = self.get_elapsed_time()

        # Stop tracking time
        super().stop_streaming()

        if self.stream is not None:
            # If a stream exists, stop it
            self.stream.stop()

            if not pause:
                # Terminate the stream
                self.stream.close()

    def run(self):
        """
        Thread execution method to continually update the audio buffer when the stream is active.
        """

        # Run the thread until it is killed
        while not self.killed:
            # Check if the microphone stream is active
            if self.query_active():
                # Determine how many samples can be read (assumed to be less than the total audio buffer size)
                num_samples_available = self.stream.read_available

                if num_samples_available > 0:
                    # Read the available samples (mono-channel) and normalize them
                    new_audio = self.stream.read(num_samples_available)[0][:, 0]

                    if self.audio_norm == -1:
                        # Perform root-mean-square normalization
                        new_audio = tools.rms_norm(new_audio)
                    else:
                        # Normalize the audio using librosa
                        new_audio = librosa.util.normalize(new_audio, norm=self.audio_norm)

                    # Advance the buffer by the amount of new samples
                    self.audio_buffer = np.roll(self.audio_buffer, -num_samples_available)
                    # Overwrite the oldest samples in the buffer with the new audio
                    self.audio_buffer[-num_samples_available:] = new_audio

                    if self.enforce_continuity:
                        # Update the pointer to the next frame start, clipping it at the oldest sample
                        self.current_sample = max(0, self.current_sample - num_samples_available)

                        if not self.suppress_warnings:
                            # Compute the current time lag
                            time_lag = (len(self.audio_buffer) - self.current_sample) / self.module.sample_rate

                            if self.current_sample == 0:
                                # Print a warning message describing the situation
                                warnings.warn(f'Processing might be too slow. Audio ' +
                                              f'buffer currently maxed out.', category=RuntimeWarning)
                            elif time_lag > MIC_LAG_TOL:
                                # Print a warning message with the current time lag
                                warnings.warn(f'Processing might be too slow. Currently ' +
                                              f'{time_lag} seconds of audio to process.', category=RuntimeWarning)
                            else:
                                # Everything is OK, no need to print anything
                                pass

    def extract_frame_features(self):
        """
        Acquire the next frame of features from the stream.

        Returns
        ----------
        features : ndarray
          Features for one frame of audio
        """

        if self.enforce_continuity:
            # Wait until there are enough new samples in the buffer for an entire frame
            while self.current_sample > len(self.audio_buffer) - self.module.get_num_samples_required():
                # This means we are ahead on processing
                continue

            # Take a full frame of audio starting at the current pointer
            audio = self.audio_buffer[self.current_sample :
                                      self.current_sample + self.module.get_num_samples_required()]
            # Update the sampler pointer to the next hop
            self.current_sample = self.current_sample + self.module.get_hop_length()
        else:
            # Simply take the most recent samples in the buffer
            audio = self.audio_buffer[-self.module.get_num_samples_required():]

        # Perform feature extraction
        features = self.module.process_audio(audio)

        return features

    def query_active(self):
        """
        Determine if the stream is currently active.

        Returns
        ----------
        active : bool
          Flag indicating the stream has been started
        """

        active = self.stream.active

        return active

    def query_finished(self):
        """
        Determine if the stream has finished.

        Returns
        ----------
        finished : bool
          Flag indicating the stream was terminated (closed)
        """

        finished = self.stream.closed

        return finished

    def get_elapsed_time(self, decimals=3):
        """
        Determine the cumulative amount of time elapsed during the
        (potentially more than one) active phase of the stream.

        Parameters
        ----------
        decimals : int
          Number of digits to keep when rounding

        Returns
        ----------
        elapsed_time : float
          Amount of time passed across while stream active
        """

        # Get the amount of time passed since the last stream start
        elapsed_time = super().get_elapsed_time()

        if self.previous_time is not None:
            # Add time when the stream was last paused
            elapsed_time += self.previous_time

        # Round the elapsed time
        elapsed_time = round(elapsed_time, decimals)

        return elapsed_time

    def stop_thread(self):
        """
        Indicate that the thread should stop running. Only call this if you are
        completely finished with the thread, since it cannot be started again.
        """

        self.killed = True

    @staticmethod
    def on_press(key):
        """
        Function called when a key is pressed.

        Parameters
        ----------
        key : pynput.keyboard.Key
          Key which was pressed
        """

        pass

    def on_release(self, key):
        """
        Function called when a key is released.

        Parameters
        ----------
        key : pynput.keyboard.Key
          Key which was released
        """

        if key == keyboard.Key.enter:
            # Stop thread execution
            self.stop_thread()
            # Wait to continue until the thread has been properly stopped
            self.join()
            # Stop the current stream
            self.stop_streaming()


class AudioStream(FeatureStream):
    """
    Implements a streaming wrapper which processes audio in real-time.
    """
    def __init__(self, module, frame_buffer_size=1, audio=None,
                 real_time=False, playback=False, suppress_warnings=True):
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
        TODO - variant where we don't automatically enforce continuity (or at least have the option here)?
        suppress_warnings : bool
          Whether to ignore warning messages
        """

        FeatureStream.__init__(self, module, frame_buffer_size)

        # Audio streaming parameters
        self.audio = None
        self.current_sample = None

        # Real-time parameters
        self.playback = playback
        self.real_time = real_time
        self.suppress_warnings = suppress_warnings

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

        # Stop streaming and clear the feature buffer
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
                if not self.suppress_warnings:
                    # Compute the current time lag
                    time_lag = self.get_elapsed_time() - sample_time

                    if time_lag > MIC_LAG_TOL:
                        # Print a warning message with the current time lag
                        warnings.warn(f'Processing might be too slow. Currently ' +
                                      f'out of sync by {time_lag} seconds.', category=RuntimeWarning)

                # Wait until it is time to acquire the next frame
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
    def __init__(self, module, frame_buffer_size=1, audio_path=None, audio_norm=-1,
                 real_time=False, playback=False, suppress_warnings=True):
        """
        Initialize parameters for the audio file streaming interface.

        Parameters
        ----------
        See AudioStream class for others...
        audio_path : string
          Path to audio to stream
        audio_norm : float or None
          Type of normalization to perform when loading audio
          -1 - root-mean-square
          See librosa for others...
            - None case is handled here
        """

        # Load the audio at the specified path, with rms normalization by default
        audio, _ = tools.load_normalize_audio(audio_path, fs=module.sample_rate, norm=audio_norm)

        self.original_audio = audio

        # Call the parent class constructor - the rest of the functionality is the same
        AudioStream.__init__(self, module, frame_buffer_size, audio, real_time, playback, suppress_warnings)

    def start_streaming(self):
        """
        Begin streaming the audio.
        """

        # Start tracking time
        super().start_streaming()

        if self.playback:
            # Play the audio
            sd.play(self.original_audio, self.module.sample_rate)
