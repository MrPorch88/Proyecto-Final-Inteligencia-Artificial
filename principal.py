import librosa
import soundfile
import numpy

def extract_feature(file_name):
    with soundfile.SoundFile(file_name) as sound_file:
        X, sample_rate = librosa.load(file_name, mono=True)
        Mfccs_mean = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=100).T, axis=0)
    return mfccs_mean