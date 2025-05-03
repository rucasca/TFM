
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
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.patches as mpatches
import matplotlib.patches as patches



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


    # Se carga la imagen de memoria en primer lugar
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




def mask_generator(coco,image_id, ids_masks : list ,path_images,  threshold = 200):

    ann_ids = coco.getAnnIds(imgIds=image_id, catIds=ids_masks, iscrowd=None)

    image_info = coco.loadImgs(image_id)[0]

    height, width = image_info['height'], image_info['width']

    img_path = os.path.join(path_images,  image_info['file_name'])
    img = Image.open(img_path).convert("RGB")

    mask  = np.zeros((height, width), dtype=np.uint8)

    if not ann_ids:
        return img, mask
    
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        if ann['area'] >= threshold:
            m = coco.annToMask(ann)
            mask = np.maximum(mask, m * ann['category_id'])
            m = coco.annToMask(ann)
            # Se supone que no se solapan nunca mascaras, en el caso de que se solapen se toma la de id mayor
            mask=np.maximum(mask,m*ann['category_id'])
   
    
    return img, mask   


# Idem pero en formato one hot encoded
def mask_generator_one_hot(coco,image_id, path_images, ids_masks : list, threshold = 200):

    img_info = coco.loadImgs(image_id)[0]

    img_path = img_info['file_name']

    img_path = os.path.join(path_images,  img_path)
    img = Image.open(img_path).convert("RGB")

    height, width = img_info['height'], img_info['width']
    num_classes = len(ids_masks)
    mask = np.zeros((height, width, num_classes + 1), dtype=np.uint8)  

    ann_ids = coco.getAnnIds(imgIds=image_id, catIds=ids_masks, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        if ann['area'] > threshold:
            class_id = ann['category_id']
            if class_id in ids_masks:
                class_index = ids_masks.index(class_id)  # position in mask channels
                m = coco.annToMask(ann)
                mask[:, :, class_index + 1] = np.maximum(mask[:, :, class_index + 1], m)

    # Set background to 1 where all other channels are 0
    mask[:, :, 0 ] = (mask[:, :, 1:].sum(axis=2) == 0).astype(np.uint8)


    return img, mask  

######  Funciones para representaciones gráficas  ########

# Representacion de una imagen y sus máscaras dado su id
# def plot_image_with_masks(image_id, masks, categories_ids, coco, images_path):
#     image_info = coco.loadImgs(image_id)[0]
#     img_path = os.path.join(images_path, image_info['file_name'])

#     original_image = cv2.imread(img_path)
#     print(img_path)
#     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#     fig , axs = plt.subplots(3, 4, figsize=(15, 10))
#     axs = axs.flatten() 
    
#     axs[0].imshow(original_image)
#     axs[0].axis('off')
#     axs[0].set_title('Image')
    
#     #print(f"We dhave {masks.shape[-1]}, {masks.shape}")
#     for idx in range(masks.shape[-1]):
        
#         # if idx >= len(axs) - 1:  
#         #     break

#         mask = masks[:,:,idx]
#         axs[idx + 1].imshow(mask)
#         axs[idx + 1].axis('off')
#         axs[idx + 1].set_title(f'{categories_ids[idx]}' if idx < len(categories_ids) else "background")
    

    
#     [fig.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]

#     plt.tight_layout()
#     plt.show()
def plot_image_and_mask(image, mask, class_id_to_name: dict):
    # Create a color map: assign a unique color for each class ID (0 is background)
    class_ids = sorted([cid for cid in np.unique(mask)])
    colors = plt.cm.get_cmap('tab10', len(class_id_to_name))  # or any other colormap

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    # Use ListedColormap to map class IDs to colors
    mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, cid in enumerate(class_ids):
        mask_colored[mask == cid] = (np.array(colors(i)[:3]) * 255).astype(np.uint8)

    ax[1].imshow(mask_colored)
    ax[1].set_title("Mask")
    ax[1].axis("off")

    # Create legend
    patches = [mpatches.Patch(color=colors(i), label=class_id_to_name[cid])
               for i, cid in enumerate(class_ids)]
    ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return

def plot_one_hot_encoded_masks(image, masks, categories_ids):

    masks = np.argmax(masks, axis=2)
    plot_image_and_mask(image, masks, categories_ids)

    return



def plot_bounding_boxes(image, result,category_info_objetive,threshold= 0.5):

    fig, ax = plt.subplots()
    ax.imshow(image)

    color_map = {cls: plt.cm.get_cmap('tab10')(i) for i, cls in enumerate(category_info_objetive.keys())}


    # Draw boxes with labels
    classess_found = []
    for box, score, label in zip(result['boxes'], result['scores'], result['labels']):
        if(label in category_info_objetive.keys() and score > threshold):
            x_min, y_min, x_max, y_max = box
            width, height = x_max - x_min, y_max - y_min
            color = color_map[label.item()]
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2,
                                    edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, f"P="+str(round(score.item(), 3)), color='white', fontsize=10,bbox=dict(facecolor=color, edgecolor='none', pad=1.5))
            classess_found.append(label)

    # Create legend
    handles = [patches.Patch(color=color_map[cls], label=category_info_objetive[cls]) for cls in category_info_objetive.keys() if cls in classess_found]
    ax.legend(handles=handles, loc='upper right')

    plt.axis('off')
    plt.tight_layout()
    plt.show()


    return