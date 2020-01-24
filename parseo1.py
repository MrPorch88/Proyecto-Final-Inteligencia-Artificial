import os
import numpy


def parseo_datos():

    path = "C:/Users/Javier Gallego/Documents/GitHub/Proyecto-Final-Inteligencia-Artificial/Data/"
    
    exclude = set(["Test1", "Test2"])
    
    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude]
        print (dirs)
        for file in files:
            if file.endswith(".WAV"):
                print(file)

parseo_datos()
