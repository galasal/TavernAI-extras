import librosa
import scipy.signal as signal
import pyrubberband

#@param filter_type: type of filter. Acceptable values are "highpass", "lowpass"
def filter_audio(audio, sr, filter_type, cutoff_freq=1000, filter_order=1):
    # Compute the filter coefficients
    nyquist_freq = 0.5 * sr
    cutoff_norm = cutoff_freq / nyquist_freq
    b, a = signal.butter(filter_order, cutoff_norm, btype=filter_type)

    # Apply the filter to the audio signal
    audio = signal.filtfilt(b, a, audio)
    audio = librosa.util.normalize(audio)
    return audio

#@param shift_amount: number of semitones to shift audio by eg. 3 means shift up by 3 semitones, -3 means shift down
def shift_frequency(audio, sr, shift_semitones):
    return pyrubberband.pitch_shift(audio, sr, n_steps=shift_semitones, rbargs={'--fine':'--fine', '--formant': '--formant'})


