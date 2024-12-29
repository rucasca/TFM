
# Fichero .py con funciones utiles auxiliares generales para cualquiera de los ficheros
import yaml
import os
import numpy as np

DIR_DATA_PREPROCESSED_TRAIN = os.path.join(
    os.path.dirname(__file__), "..", "data", "preprocessed_train")


# FUNCIONES RELATIVAS A LAS CONTRANTES
def load_yaml_file() -> dict:

    """

    Funcion que caraga en fichero yml y lo devuelve como un diccionario nativo de Python

    """
    
    path =os.path.join(os.path.dirname(__file__), r"config.yml")

    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
            return yaml.safe_load(file)
    


# FUNCIONES RELATIVAS A LA CARGA/ PREPROCESAMIENTO DE DATOS
def save_numpy_array(np_array, nombre):
    """
    Almacena un numpy array n dimensional como.npz.

    Args:
        np_array (tuple): imagen a almacenar.
        nombre (str): nombre del archivo.
    """  

    np_array_image, np_array_mask = np_array[0], np_array[1]

    dir = os.path.join(DIR_DATA_PREPROCESSED_TRAIN, nombre)
      
    # Lo cargamos con el compressed al contener mascaras, dado qeu son datos de caracter repetitivo
    np.savez_compressed(dir, image=np_array_image, mask =np_array_mask )

    return
      

