
# Fichero .py con funciones utiles auxiliares generales para cualquiera de los ficheros
from matplotlib.figure import Figure
import yaml
import os
import numpy as np
import requests
import zipfile
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
from matplotlib.patches import Patch



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
    destination_folder = os.path.join(os.path.dirname(__file__), destination_folder)
    
    os.makedirs(destination_folder, exist_ok=True)
    zip_path = os.path.join(destination_folder, zip_filename)
    
    # Lanza la petición
    response = requests.get(url)

    
    # Si devuelve 200 (exito), guarda en memoria en formato .zip
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            print(f"Fichero {zip_path} encontrado desde {url}")
            f.write(response.content)

        print("Fichero descargado")

        # Hace unzip del resultado
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            

            zip_ref.extractall(destination_folder)
            print("Fichero descomprimido\n")

        os.remove(zip_path)

        print(f"ZIP file downloaded to: {zip_path}")
    else:
        print(f"fallo en la descarga del fichero: {response.status_code}")

    return 



# Relativas a operaciones repetidas a lo largo del codigo para la ordenación de elementos
def top_n_values(input_dict, n):
    """
    Devuelve un diccionario con las n claves que tienen los valores más altos.
    
    Args:
        input_dict (dict): El diccionario original.
        n (int): El número de pares clave-valor a incluir en el nuevo diccionario.
        
    Returns:
        dict: Un diccionario con las n claves de mayor valor.
    """
    if n <= 0:
        return {}
    
    # ordenación las claves del diccionario según sus valores, en orden descendente
    sorted_keys = sorted(input_dict, key=input_dict.get, reverse=True)
    
    top_keys = sorted_keys[:n]

    top_dict = {key: input_dict[key] for key in top_keys}
    
    return top_dict 


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
      




# Funciones relativas a el printeo de mascaras
# 1) Para la el ground truth de la imagen
def plot_masks_given_id_image(id_image:int, coco:COCO, yaml_file:dict) -> Figure:

    """Plotea la mascara de una imagen y la imagen dado su id"""


    DIR_TRAIN_IMGS = yaml_file["dirs"]["imagenes"]["train"]
    DIR_TRAIN_IMGS = os.path.join(os.path.dirname(__file__),"..", DIR_TRAIN_IMGS)


    # Cargo la imagen de memoria en primer lugar
    image = coco.loadImgs(id_image)[0]
    img_path = os.path.join(DIR_TRAIN_IMGS, image['file_name'])
    annotation_ids = coco.getAnnIds(imgIds=id_image)
    annotations = coco.loadAnns(annotation_ids)

    # Carga de los nombres de las categorias
    categories = coco.loadCats(coco.getCatIds())
    category_id_to_name = {category['id']: category['name'] for category in categories}

    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    fig , axs = plt.subplots(2,1, figsize=(15, 10))
    axs = axs.flatten()  

    axs[0].imshow(original_image)
    axs[0].axis('off')
    axs[0].set_title('Imagen original')

    legend_elements = []
    # print(type(image))
    # print(image)

    matriz_base = np.zeros((*original_image.shape[:2], 3), dtype=np.uint8)

    # Generamos colores aleatorios para las mascaras

    # Garantizamos la reproducibilidad fijando la semilla
    np.random.seed(42) 
    colors = np.random.randint(0, 256, size=(len(annotations), 3))
    anotated = []
    colors_anotated = {}

    # Dibujar cada máscara con un color único
    for ann, color in zip(annotations, colors[:len(annotations)]):
        label = ann['category_id'] 
        mask = coco.annToMask(ann)

        if(label in anotated):
            color = colors_anotated[label]
        else:
            colors_anotated[label] = color
            anotated.append(label)
            legend_elements.append(Patch(facecolor=np.array(color) / 255, label=f'{category_id_to_name[label]}'))

        for c in range(3):  
            matriz_base[:, :, c] += mask * color[c]

    axs[1].imshow(matriz_base)
    axs[1].axis('off')
    axs[1].set_title('Máscara')
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.5, 1), title="Clase")
    plt.tight_layout()
    plt.show()

    return fig


# yaml_file = load_yaml_file()
# DIR_TRAIN_ANNOTATIONS = yaml_file["dirs"]["anotaciones"]["train"]
# DIR_TRAIN_IMGS = yaml_file["dirs"]["imagenes"]["train"]
# DIR_TRAIN_IMGS = os.path.join(os.getcwd(),"..", DIR_TRAIN_IMGS)

# coco=COCO(os.path.join(os.path.dirname(__file__),"..", DIR_TRAIN_ANNOTATIONS))
# fig = plot_masks_given_id_image(118113, coco, yaml_file)
# plt.show()

# 2) Para la máscara predicha


