import numpy as np
import librosa
import sounddevice as sf

"""def modify_pitch_and_speed(audio, pitch_factor, speed_factor):
    y_pitch = librosa.effects.pitch_shift(audio[:, 0], sr=Fs, n_steps=pitch_factor - 1)
    y_speed = librosa.effects.time_stretch(y_pitch, rate=1/speed_factor)
    modified_audio = np.column_stack((y_speed,))

    return modified_audio"""

def modify_speed(audio, speed_factor):
    y_speed = librosa.effects.time_stretch(audio, rate=1/speed_factor)
    modified_audio = np.column_stack((y_speed,))
    return modified_audio

def modify_speed_and_timbre(audio, speed_factor, timbre_factor):
    # Time stretching
    y_speed = librosa.effects.time_stretch(audio, rate=1/speed_factor)
    # Timbre modification (preemphasis)
    y_timbre = librosa.effects.preemphasis(y_speed, coef=timbre_factor)
    modified_audio = np.column_stack((y_timbre,))
    return modified_audio

# Define audio parameters
Fs = 44100  # Sampling frequency
duration = 10  # Recording duration in seconds

# Record audio
print('Start speaking...')
y = sf.rec(int(duration * Fs), samplerate=Fs, channels=1, dtype='float64')
sf.wait()
print('Recording stopped.')

# Extract audio data
y = y.flatten()

# Define the pitch shifting factor (adjust for desired effect)
pitch_factor = 0.85 # Example: Changing the pitch to make it lower
speed_factor = 0.8  # Adjust as needed
timbre_factor = 100

# Perform time-scale modification (pitch shifting)
y_modified = np.interp(np.arange(0, len(y), pitch_factor), np.arange(len(y)), y).astype('float32')

# Perform speed modification
# enhanced_audio = modify_speed(y_modified, speed_factor)
enhanced_audio = modify_speed_and_timbre(y_modified.flatten(), speed_factor, timbre_factor)

# Play the enhanced audio (optional)
print("Playing processed audio....")
sf.play(enhanced_audio.flatten(), Fs)  # Use flattened version of enhanced_audio
sf.wait()