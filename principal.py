import librosa
import numpy as np
import soundfile as sf
import pyaudio
import os
from sklearn.naive_bayes import CategoricalNB
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
       
    with sf.SoundFile(file_name) as sound_file:
        X, sample_rate = librosa.load(file_name, mono=True)
        mfccs_mean = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=100).T, axis=0)
        #mfcc_delta = librosa.feature.delta(mfccs)
        #mfcc_delta_delta = librosa.feature.delta(mfcss, order=2)
    return mfccs_mean

def bayes(train_X, train_Y, test_X, test_Y):
    

if __name__ == '__main__':
    #file = 'Proyecto-Final-Inteligencia-Artificial/Data/0/Alegria_1.WAV'
    file = 'Data/0/Alegria_1.WAV'
    print(extract_feature(file))

