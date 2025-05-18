
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
    destination_folder   =    os.path.join(os.path.dirname(__file__),destination_folder)
    
    os.makedirs(destination_folder,exist_ok=True)
    zip_path = os.path.join(destination_folder,zip_filename)
    
    response = requests.get(url)

    # La peticion puede devolver 200, en el caso exitoso, cualquier otro valor indica redireccion, fallo, etc y se ignora
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            print(f"Fichero {zip_path} encontrado desde {url}")
            f.write(response.content)

        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            
            # Print descompresion del fichero
            zip_ref.extractall(destination_folder)
            print("Fichero descomprimido\n")

        os.remove(zip_path)

        print(f"ZIP file descargaod en {zip_path}")
    else:
        print(f"fallo en la descarga del fichero: {response.status_code}")

    return 



# Relativas a operaciones repetidas a lo largo del codigo para la ordenación de elementos

# Esta primera empleada en la obtencion de las n claves con mayor value, empleada en varias ocasiones en el AED
def top_n_values(input_dict:dict, n: int) -> dict :
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


# Esta segunda guarda un np array, en este caso imagenes, en formato npz para un menor uso de memoria en el entrenamiento de modelos
def save_numpy_array(np_array, nombre):
    """
    Almacena un numpy array n dimensional como.npz.
    Args:
        np_array: imagen a almacenar
        nombre: nombre del archivo

    """  

    np_array_image, np_array_mask = np_array[0], np_array[1]

    dir = os.path.join(DIR_DATA_PREPROCESSED_TRAIN, nombre)
    np.savez_compressed(dir, image=np_array_image, mask =np_array_mask )

    return
      


#### TODO: generar el docstring de las funciones restantes


############    FUNCIONES DE PLOTTING    ###############

# Funcion apra visualizar el ground truth de la imagen dado su id
def plot_masks_given_id_image(id_image:int, coco:COCO, yaml_file:dict) -> Figure:

    DIR_TRAIN_IMGS = yaml_file["dirs"]["imagenes"]["train"]
    # Necesario en este caso por estar contenido en un .py file en vez de un .ipynb ya que las rutas funcionan de forma distinta
    DIR_TRAIN_IMGS = os.path.join(os.path.dirname(__file__),"..", DIR_TRAIN_IMGS)


    image = coco.loadImgs(id_image)[0]
    img_path = os.path.join(DIR_TRAIN_IMGS, image['file_name'])
    annotation_ids = coco.getAnnIds(imgIds=id_image)
    annotations = coco.loadAnns(annotation_ids)

    categories = coco.loadCats(coco.getCatIds())
    category_id_to_name = {category['id']: category['name'] for category in categories}
    #print("categories are", category_id_to_name)

    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)



    fig,axs = plt.subplots(2,1, figsize=(15, 10))
    axs= axs.flatten()  
    axs[0].imshow(original_image)
    axs[0].axis('off')
    axs[0].set_title('Imagen original')

    legend_elements = []
    # print(type(image))
    # print(image)

    # Se inicializa todo como fondo
    matriz_base = np.zeros((*original_image.shape[:2], 3), dtype=np.uint8)

    # Se generan colores aleatorios para cada clase
    np.random.seed(42) 
    colors = np.random.randint(0, 256, size=(len(annotations), 3))
    anotated = []
    colors_anotated = {}

    # para cada máscara encontrada se sustituye su key de máscara en la matriz base
    # cabe destacar (comprobado en el cto de datos) que no existen pixeles con más de 1 clase
    for ann, color in zip(annotations, colors[:len(annotations)]):
        label = ann['category_id'] 
        mask = coco.annToMask(ann)

        if(label in anotated):
            color = colors_anotated[label]
        else:
            colors_anotated[label] = color
            anotated.append(label)
            # Se añade la clase encontrada a la leyenda, dado qeu se omiten las categorías no presentes
            legend_elements.append(Patch(facecolor=np.array(color) / 255, label=f'{category_id_to_name[label]}'))

        for c in range(3):  
            matriz_base[:,:,c]+=mask*color[c]

    axs[1].imshow(matriz_base)
    axs[1].axis('off')
    axs[1].set_title('Máscara')
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.5,1),title="Clase")
    plt.tight_layout()
    plt.show()

    return fig



# Función para la generación de la máscara de una imagen en formaro 3 canales
def mask_generator(coco,image_id, ids_masks : list ,path_images,  threshold = 200):

    ann_ids = coco.getAnnIds(imgIds=image_id,catIds=ids_masks,iscrowd=None)
    image_info = coco.loadImgs(image_id)[0]
    height, width = image_info['height'],image_info['width']
    img_path = os.path.join(path_images,  image_info['file_name'])
    img = Image.open(img_path).convert("RGB")

    # Mascara con todo fondo como base
    mask  = np.zeros((height, width), dtype=np.uint8)
    if not ann_ids:
        return img, mask
    
    anns = coco.loadAnns(ann_ids)
    # Se itera por máscara etiquetada, si supera el tamaño del umbral (se ignoran máscaras pequeñas)
    for ann in anns:
        if ann['area'] >= threshold:
            m= coco.annToMask(ann)
            mask = np.maximum(mask,m*ann['category_id'])
            m =coco.annToMask(ann)
            # Se supone que no se solapan nunca mascaras, en el caso de que se solapen se toma la de id mayor (no sucede en este cto de datos)
            mask=np.maximum(mask,m*ann['category_id'])
   
    
    return img, mask   


# Idem pero en formato one hot encoded con N canales
def mask_generator_one_hot(coco,image_id, path_images, ids_masks : list, threshold = 200):

    img_info =coco.loadImgs(image_id)[0]

    img_path = img_info['file_name']

    img_path = os.path.join(path_images,  img_path)
    img = Image.open(img_path).convert("RGB")

    height,width= img_info['height'], img_info['width']
    num_classes= len(ids_masks)
    mask= np.zeros((height,width,num_classes +1),dtype=np.uint8)  

    ann_ids= coco.getAnnIds(imgIds=image_id,catIds=ids_masks,iscrowd=None)
    anns= coco.loadAnns(ann_ids)

    for ann in anns:

        # Omision de las mascaras que no superen un threshold de pixeles dado
        if ann['area']> threshold:
            class_id =ann['category_id']
            if class_id in ids_masks:
                class_index =ids_masks.index(class_id) 
                m =coco.annToMask(ann)
                mask[:,:,class_index+1] = np.maximum(mask[:,:,class_index+1],m)

    mask[:,:,0] = (mask[:,:,1:].sum(axis=2)==0).astype(np.uint8)


    return img,mask  

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



# Ploteo de imágenes junto con su mascara
def plot_image_and_mask(image,mask,class_id_to_name: dict):
    class_ids = sorted([cid for cid in np.unique(mask)])

    # Uso de templates de colores predefinidas en matplot para que exista mas contraste entre clases de ids adyacentes, usando tab10
    colors = plt.cm.get_cmap('tab10',len(class_id_to_name))  

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Imagen")
    ax[0].axis("off")

    # Se incializa la mascara como todo fondo, y se itera añadiendo 
    mask_colored = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
    for i,cid in enumerate(class_ids):
        mask_colored[mask==cid] = (np.array(colors(i)[:3])*255).astype(np.uint8)

    # print(mask_colored)
    ax[1].imshow(mask_colored)
    ax[1].set_title("Máscara")
    ax[1].axis("off")

    # generación de la leyenda
    patches = [mpatches.Patch(color=colors(i), label=class_id_to_name[cid])
               for i, cid in enumerate(class_ids)]
    ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return

# Automatizacion en el formato one hot encoded a partir de la anterior función
def plot_one_hot_encoded_masks(image, masks, categories_ids):

    masks = np.argmax(masks, axis=2)
    plot_image_and_mask(image, masks, categories_ids)

    return


# Automatizacion en el formato one hot encoded a partir de la anterior función, en este caso para inputs que el canal de las cases sea el 0 en lugar del 2
def plot_one_hot_encoded_masks_norm(image, masks, categories_ids):

    masks = np.argmax(masks, axis=0)
    plot_image_and_mask(image, masks, categories_ids)

    return



# Representación de N  imagenes y sus máscaras de forma simultanea
def plot_bounding_boxes(images, results, category_info_objetive, threshold=0.5):

    # Calculo del numero de filas necesarias (se colocan 2 por fila)
    n = len(images)
    cols=2
    rows=(n+1)//cols

    fig,axes = plt.subplots(rows, cols, figsize=(8*cols,6*rows))
    axes = axes.flatten() if n > 1 else [axes]

    id_objetives=category_info_objetive.keys()

    # idem que en las otras funciones, uso de un cto de colores predeterminado
    color_map={cls: plt.cm.get_cmap('tab10')(i) for i, cls in enumerate(id_objetives)}

    for idx,(image,result) in enumerate(zip(images,results)):
        ax = axes[idx]
        ax.imshow(image)
        classess_found = []

        for box,score,label in zip(result['boxes'],result['scores'],result['labels']):
            if label.item() in id_objetives and score>threshold:
                #print("nueva clase que supera el umbral definido", label)
                x_min, y_min, x_max, y_max = box
                width, height = x_max - x_min, y_max - y_min
                color = color_map[label.item()]
                rect = patches.Rectangle((x_min, y_min),width,height,linewidth=2,
                                         edgecolor=color,facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min-10,f"P={score.item():.3f}",color='white',fontsize=10,
                        bbox=dict(facecolor=color,edgecolor='none',pad=1.5))
                classess_found.append(label)

        handles=[mpatches.Patch(color=color_map[cls], label=category_info_objetive[cls])
                   for cls in id_objetives if cls in classess_found]
        ax.legend(handles=handles, loc='upper right')
        ax.axis('off')

    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return


# Idem para la anterior pero usando EEDD diferentes a tensores en los parametros
def plot_bounding_boxes_non_tensor(images, results, category_info_objetive, threshold=0.5):

    # Calculo del numero de filas necesarias (se colocan 2 por fila)
    n = len(images)
    cols=2
    rows=(n+1)//cols

    fig,axes = plt.subplots(rows, cols, figsize=(8*cols,6*rows))
    axes = axes.flatten() #if n > 1 else [axes]

    id_objetives=category_info_objetive.keys()

    # idem que en las otras funciones, uso de un cto de colores predeterminado
    color_map={cls: plt.cm.get_cmap('tab10')(i) for i, cls in enumerate(id_objetives)}
    
    for idx,(image,result) in enumerate(zip(images,results)):
        ax = axes[idx]
        ax.imshow(image)
        classess_found = []

        for box,score,label in zip(result['boxes'],result['scores'],result['labels']):
            if label in id_objetives and score>threshold:
                #print("nueva clase que supera el umbral definido", label)
                x_min, y_min, x_max, y_max = box
                width, height = x_max - x_min, y_max - y_min
                color = color_map[label]
                rect = patches.Rectangle((x_min, y_min),width,height,linewidth=2,
                                         edgecolor=color,facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min-10,f"P={score:.3f}",color='white',fontsize=10,
                        bbox=dict(facecolor=color,edgecolor='none',pad=1.5))
                classess_found.append(label)

        handles=[mpatches.Patch(color=color_map[cls], label=category_info_objetive[cls])
                   for cls in id_objetives if cls in classess_found]
        ax.legend(handles=handles, loc='upper right')
        ax.axis('off')

    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return

# def plot_bounding_boxes(image, result,category_info_objetive,threshold= 0.5):

#     fig, ax = plt.subplots()
#     ax.imshow(image)

#     color_map = {cls: plt.cm.get_cmap('tab10')(i) for i, cls in enumerate(category_info_objetive.keys())}


#     classess_found = []
#     for box, score, label in zip(result['boxes'], result['scores'], result['labels']):
#         #print("checking acc", score)
#         if(label.item() in category_info_objetive.keys() and score > threshold):
            
#             x_min, y_min, x_max, y_max = box
#             width, height = x_max - x_min, y_max - y_min
#             color = color_map[label.item()]
#             rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2,
#                                     edgecolor=color, facecolor='none')
#             ax.add_patch(rect)
#             ax.text(x_min, y_min - 10, f"P="+str(round(score.item(), 3)), color='white', fontsize=10,bbox=dict(facecolor=color, edgecolor='none', pad=1.5))
#             classess_found.append(label)

#     handles = [patches.Patch(color=color_map[cls], label=category_info_objetive[cls]) for cls in category_info_objetive.keys() if cls in classess_found]
#     ax.legend(handles=handles, loc='upper right')

#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()


#     return


#######    PLOTTINGS DE COMPARATIVA DE RESULTADOS     ########


# Diferencias entre imagen real, mascara predicha y ground truth
def plot_differences(image, mask_gt, mask_predicted,class_id_to_name):
    class_ids = sorted([cid for cid in np.unique(mask_gt)])
    colors = plt.cm.get_cmap('tab10', len(class_id_to_name))

    # Se distribuye el plot en 1 unica linea de 3 columnas
    fig, ax = plt.subplots(1,3, figsize=(15,6))
    ax[0].imshow(image)
    ax[0].set_title("Imagen")
    ax[0].axis("off")


    # Inicializacion del fondo
    mask_colored = np.zeros((mask_gt.shape[0], mask_gt.shape[1], 3), dtype=np.uint8)
    for i, cid in enumerate(class_ids):
        mask_colored[mask_gt == cid] = (np.array(colors(i)[:3]) * 255).astype(np.uint8)

    ax[1].imshow(mask_colored)
    ax[1].set_title("Ground truth")
    ax[1].axis("off")

    # Representacion de la mascara predicha
    mask_colored = np.zeros((mask_predicted.shape[0],mask_predicted.shape[1],3),dtype=np.uint8)
    for i, cid in enumerate(class_ids):
        mask_colored[mask_predicted==cid]=(np.array(colors(i)[:3])*255).astype(np.uint8)

    ax[2].imshow(mask_colored)
    ax[2].set_title("Máscara predicha")
    ax[2].axis("off")

    handles = [mpatches.Patch(color=colors(i), label=class_id_to_name[cid])
               for i, cid in enumerate(class_ids)]
    ax[2].legend(handles=handles,bbox_to_anchor=(1.050,1),loc='upper left')

    plt.tight_layout()
    plt.show()
    return 


# Idem qeu la superior, pero con n imagenes
def plot_differences_batch(images, masks_gt, masks_predicted, class_id_to_name):

    # tantas filas como imagenes, 3 columnas en total como anteriormente
    n = len(images)
    
    all_class_ids = sorted({cid for mask in masks_gt for cid in np.unique(mask)})
    colors = plt.cm.get_cmap('tab10',len(class_id_to_name))  
    
    fig, axes = plt.subplots(n,3,figsize=(15,5*n))
    if n == 1:
        axes = np.expand_dims(axes,0)  # Ensure axes is 2D for consistency

    for idx in range(n):

        
        image,mask_gt,mask_pred = images[idx],masks_gt[idx],masks_predicted[idx]
        present_masks_image = sorted(np.unique(np.concatenate((mask_gt,mask_pred))))
        
        axes[idx,0].imshow(image)
        axes[idx,0].set_title(f"Imagen {idx}")
        axes[idx,0].axis("off")
        mask_colored_gt=np.zeros((*mask_gt.shape,3),dtype=np.uint8)
        for i,cid in enumerate(present_masks_image):
            mask_colored_gt[mask_gt==cid]=(np.array(colors(i)[:3])*255).astype(np.uint8)
        
        axes[idx,1].imshow(mask_colored_gt)
        axes[idx,1].set_title("Ground Truth")
        axes[idx,1].axis("off")
        
        mask_colored_pred=np.zeros((*mask_pred.shape,3),dtype=np.uint8)
        for i,cid in enumerate(present_masks_image):
            mask_colored_pred[mask_pred==cid] =(np.array(colors(i)[:3])*255).astype(np.uint8)
        
        axes[idx,2].imshow(mask_colored_pred)
        axes[idx,2].set_title("Máscaras predichas")
        axes[idx,2].axis("off")

        handles=[mpatches.Patch(color=colors(i), label=class_id_to_name[cid])
                   for i, cid in enumerate(present_masks_image)]
        axes[idx,2].legend(handles=handles,bbox_to_anchor=(1.052, 1),loc='upper left')
    

    
    plt.tight_layout()
    plt.show()

    return


def plot_pipeline_yolo_sam_batch(images, masks_gt, masks_predicted, categories_names_by_index):

    # tantas filas como imagenes, 3 columnas en total como anteriormente
    n = len(images)
    
    colors = plt.cm.get_cmap('tab10',len(categories_names_by_index.keys()))  
    print("colors are ", colors)
    print("cats are ", categories_names_by_index)
    
    fig, axes = plt.subplots(n,3,figsize=(15,5*n))
    if n == 1:
        axes = np.expand_dims(axes,0)  # Ensure axes is 2D for consistency

    for idx in range(n):

        
        image,mask_gt,mask_pred =images [idx],masks_gt[idx],masks_predicted[idx]
        present_masks_image = sorted(np.union1d(masks_gt[idx],masks_predicted[idx]))
        
        axes[idx,0].imshow(image)
        axes[idx,0].set_title(f"Imagen {idx}")
        axes[idx,0].axis("off")


        mask_colored_gt=np.zeros(image.shape,dtype=np.uint8)
        for i,cid in enumerate(sorted(categories_names_by_index.keys())):
            mask_colored_gt[mask_gt==cid]=(np.array(colors(i)[:3])*255).astype(np.uint8)
        
        axes[idx,1].imshow(mask_colored_gt)
        axes[idx,1].set_title("Ground Truth")
        axes[idx,1].axis("off")
        
        mask_colored_pred=np.zeros((*mask_pred.shape,3),dtype=np.uint8)
        for i,cid in enumerate(sorted(categories_names_by_index.keys())):
            mask_colored_pred[mask_pred==cid] =(np.array(colors(i)[:3])*255).astype(np.uint8)
        
        axes[idx,2].imshow(mask_colored_pred)
        axes[idx,2].set_title("Máscaras predichas")
        axes[idx,2].axis("off")

        handles=[mpatches.Patch(color=colors(cid), label=categories_names_by_index[cid])
                   for i, cid in enumerate(present_masks_image)]
        axes[idx,2].legend(handles=handles,bbox_to_anchor=(1.052, 1),loc='upper left')
    

    
    plt.tight_layout()
    plt.show()

    return


# Encoder de la mascara para obtener su formato one hot encoded

def one_hot_encoder_masks(mask, category_info_objective):
    target_classes=sorted(category_info_objective.keys())
    one_hot_mask=np.zeros((len(target_classes), mask.shape[0], mask.shape[1]), dtype=np.uint8)
    
    for i, id_class in enumerate(target_classes):
        one_hot_mask[i,:,:] = (mask==id_class).astype(np.uint8)
    
    return one_hot_mask