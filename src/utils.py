
# Fichero .py con funciones utiles auxiliares generales para cualquiera de los ficheros
import yaml
import os
import numpy as np
import requests
import zipfile



DIR_DATA_PREPROCESSED_TRAIN = os.path.join(
    os.path.dirname(__file__), "..", "data", "preprocessed_train")


# FUNCIONES RELATIVAS A LAS GESTION DE FICHEROS
def load_yaml_file() -> dict:

    """

    Funcion que caraga en fichero yml y lo devuelve como un diccionario nativo de Python

    """
    
    path =os.path.join(os.path.dirname(__file__),"..", r"config.yml")

    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
            return yaml.safe_load(file)
    


# DESCARGA DE UN FICHERO ZIP   
def download_zip(url:str, destination_folder:str, zip_filename:str):

    """Carga de un zip de internet"""
    os.makedirs(destination_folder, exist_ok=True)
    zip_path = os.path.join(destination_folder, zip_filename)
    
    # Lanza la petición
    response = requests.get(url)
    
    # Si devuelve 200 (exito), guarda en memoria en formato .zip
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            print("Fichero encontrado")
            f.write(response.content)

        print("fichero descargado")

        # Hace unzip del resultado
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("fichero descompruimido")

            zip_ref.extractall(destination_folder)

        os.remove(zip_path)

        print(f"ZIP file downloaded to: {zip_path}")
    else:
        print(f"fallo en la descarga del fichero: {response.status_code}")

    return 




    


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
      

