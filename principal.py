import librosa
import numpy as np
import soundfile
import pyaudio
import os
import sklearn
#[
# {
# ruta:'ruta',
# emocion: 'alegria'
# },
# ruta:'ruta',
# emocion: 'tristeza'
# },etc etc
# ]
# Metodo de parseo de datos
def parseo_datos():
    entradas = os.listdir('Data/')
    for entrada in entradas:
        print(entrada) # Deberia de imprimir el nombre de cada archivo de la carpeta principal de audios



# Metodo para seleccionar archivos de audio


# Metodo para extraer las caracteristicas del audio en cuestion. MFCCs
def extract_feature(file_name):
    with soundfile.SoundFile(file_name) as sound_file:
        X, sample_rate = librosa.load(file_name, mono=True)
        mfccs_mean = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=100).T, axis=0)
    return mfccs_mean
