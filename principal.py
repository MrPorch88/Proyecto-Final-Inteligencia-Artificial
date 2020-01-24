import librosa
import numpy as np
import soundfile as sf
import pyaudio
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Metodo para seleccionar archivos de audio
def parseo_datos(direc):
    X = []
    y = []

    for root, dirs, files in os.walk(direc):
        for name in files:
            emotion = name.split('_')
            if not root.endswith('Test1') and not root.endswith('Test2'):
                try:
                    features = extract_feature( root + '/' + name)
                    X.append(features)
                    y.append(emotion[0])
                except:
                    print(root + '/' + name + ' - Falla')
                    continue

    return train_test_split(np.array(X), y)

# Metodo para extraer las caracteristicas del audio en cuestion. MFCCs
def extract_feature(file_name):
       
    with sf.SoundFile(file_name) as sound_file:
        X, sample_rate = librosa.load(file_name, mono=True)
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=100).T
        mfccs_mean = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=100).T, axis=0)
        mfcc_delta = np.mean(librosa.feature.delta(mfccs, width=3).T, axis=1)
        mfcc_delta_delta = np.mean(librosa.feature.delta(mfccs, order=2).T, axis=1)

        mfccs = np.hstack((mfccs_mean, mfcc_delta, mfcc_delta_delta))
    return mfccs

# Metodo de Redes neuronales con los argumentos:
# hidden_layer_sizes=(100,), verbose=True, max_iter=2000, learning_rate='constant'
def neuralNetwork100_0(X_train, X_test, y_train, y_test):
    neuralNetwork = MLPClassifier(hidden_layer_sizes=(100,100,100), verbose=True, max_iter=2000, learning_rate='adaptive')
    neuralNetwork.fit(X_train, y_train)
    y_pred = neuralNetwork.predict(X_test)
    print("Neural Network. Argumentos: hidden_layer_sizes=(100,), verbose=True, max_iter=2000, learning_rate='constant'")
    print(neuralNetwork.get_params)
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


# Metodo de Bayes 
def naiveBayes(X_train, X_test, y_train, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Gaussian Naive Bayes")
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

def ejecucionModelos(X_train, X_test, y_train, y_test):
    #naiveBayes(X_train, X_test, y_train, y_test)
    neuralNetwork100_0(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    #file = 'Proyecto-Final-Inteligencia-Artificial/Data/0/Alegria_1.WAV'
    file = 'Data/0/Asco_1.WAV'
    direc = 'Data'
    X_entre, X_testeo, y_entre, y_testeo = parseo_datos(direc)
    ejecucionModelos(X_entre, X_testeo, y_entre, y_testeo)
