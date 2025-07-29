
# import numpy as np
# from segment_anything import SamAutomaticMaskGenerator
# import pandas as pd
# import torch
# import os


# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO

# from typing import Literal
# from skimage.transform import resize
# from tqdm import tqdm

# import time
# from pathlib import Path
# import os
# from transformers import CLIPProcessor, CLIPModel

# import matplotlib.pyplot as plt
# import cv2
# from pycocotools.coco import COCO
# import numpy as np
# import random

# from pathlib import Path
# import os

# from clip import load, tokenize
# import torch
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# import time

# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# from segment_anything import SamAutomaticMaskGenerator

# import clip
# import pandas as pd
# from sklearn.metrics import average_precision_score

# from tqdm import tqdm

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import matplotlib.patches as mpatches
# import tensorflow as tf
# import numpy as np

# from ultralytics import YOLO
# from segment_anything import sam_model_registry, SamPredictor

# from torchvision.models.detection import retinanet_resnet50_fpn

# import torchvision.transforms as T
# from tensorflow import keras

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# import json

# from dotenv import load_dotenv
# import os
# from pathlib import Path


# #### CONST

# env_path = Path(__file__).resolve().parents[1] / '.env'
# load_dotenv(dotenv_path=env_path)
# DIR_SAM = os.getenv('DIR_FAST_SAM')
# DIR_YOLO = os.getenv('DIR_YOLO')
# BASE_MODEL_UNET = os.getenv('BASE_MODEL_UNET')
# FINAL_MODEL_UNET = os.getenv('FINAL_MODEL_UNET')


# DIR_CONSTANTS = r"inputs\constants.json"

# with open(DIR_CONSTANTS, 'r') as file:
#     CONSTANTS =json.load(file)
#     CONSTANTS["cons_threshold"] = 0.5


# OBJECTIVES = CONSTANTS["objetives"]
# CATEGORIES = CONSTANTS["categories"]

# ID_OBJECTIVES = CONSTANTS["id_objetives"]
# CATEGORY_INFO_OBJECTIVE = CONSTANTS["category_info_objetive"]

# DICT_CLASS_INDEX = CONSTANTS["dict_class_index"]
# CONS_TRHESHOLD = CONSTANTS["cons_threshold"]




# #### INFERENCE FUNCTIONS
# def process_inference_base_unet(img,unet_model_baseline):

#     # Preprocesmiento para que encaje con el modelo (entrenado con shape 256x256 comun a todas las imagenes)
#     target_size = (256, 256)
#     img_resized = cv2.resize(img, target_size) 
#     img_resized = img_resized.astype(np.float32) / 255.0
#     # Se añade la dimension del batch para poder aplicar la inferencia
#     input_tensor = np.expand_dims(img_resized,axis=0)  

#     # Ejecucion de la inferencia
#     prediction = unet_model_baseline.predict(input_tensor)  

#     # Postprocesamiento
#     pred_mask = np.argmax(prediction[0],axis=-1) 
#     prediction  = prediction[0]

#     return pred_mask,prediction 




# def yolov8_inference(image, yolo_model):

#     yolo_boxes = []
#     yolo_prob = []
#     yolo_label = []

#     results = yolo_model(image)

#     for box_info in results[0]:
#         x1,y1,x2,y2, conf, label = box_info.boxes.data[0].tolist()

#         # Dado que yolo detecta más clases que nuestro conjunto de objetivos presente, filtramos las bounding boxes encontradas
#         if (label+1) in ID_OBJECTIVES:
#             #x1,y1,x2,y2 = map(int, box.xyxy[0])
#             yolo_boxes.append( (x1,y1,x2,y2))
#             yolo_prob.append(conf)
#             yolo_label.append(label + 1)

#     return yolo_boxes, yolo_prob, yolo_label


# def sam_segmentation_object_detection_yolo(image,yolo_boxes, yolo_probs, yolo_labels,sam_model, DICT_CLASS_INDEX ):
#     sam_model.set_image(image)

#     final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     current_scores = np.zeros(image.shape[:2], dtype=np.uint8)

#     for box, score, label in zip(yolo_boxes, yolo_probs, yolo_labels):
#         if(score > CONS_TRHESHOLD and label in CATEGORY_INFO_OBJECTIVE.keys()):
#             masks, scores, logits = sam_model.predict(
#                 point_coords=None,
#                 point_labels=None,
#                 box= np.array([round(x) for x in box]),
#                 multimask_output=False
#             )

#             if np.any(masks):

#                 mask_values = np.where(masks[0], scores, 0)
#                 final_mask = np.where(mask_values > current_scores , label, final_mask)
#                 current_scores = np.maximum(mask_values, current_scores)


#     lookup = np.vectorize(lambda k: DICT_CLASS_INDEX[k])
#     mask_by_index = lookup(final_mask)

#     return mask_by_index, current_scores



# def process_inference_yolo_sam(image,yolo_model,sam_model,DICT_CLASS_INDEX ):

#     yolo_boxes, yolo_prob, yolo_label = yolov8_inference(image,yolo_model)

#     mask, score = sam_segmentation_object_detection_yolo(image,yolo_boxes, yolo_prob, yolo_label, sam_model, DICT_CLASS_INDEX )

#     score = np.where(mask == 0, 1, score)

#     score_preprocessed  = np.zeros((score.shape[0], score.shape[1], len(DICT_CLASS_INDEX.keys())))

#     for _, val in DICT_CLASS_INDEX.items():
#         score_preprocessed[:,:,val] = np.where(mask == val, score, 0 )

#     score_preprocessed[:,:,0] = 1-np.sum(score_preprocessed[:,:,1:],axis=-1)

#     return mask,  score_preprocessed




# def retina_inference(image, retina_model):
    
#     transform = T.Compose([
#         T.ToTensor()
#     ])
#     img_tensor = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         outputs = retina_model(img_tensor)

#     boxes = outputs[0]['boxes']
#     scores = outputs[0]['scores']
#     labels = outputs[0]['labels']

#     return boxes, scores, labels



# def sam_segmentation_object_detection_retina(image,retina_boxes, retina_prob, retina_label,sam_model, DICT_CLASS_INDEX ):
#     sam_model.set_image(image)

#     final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     current_scores = np.zeros(image.shape[:2], dtype=np.uint8)

#     # print("having label ", retina_label, len(retina_label))
#     # print("having retina_prob ", len(retina_prob))
#     # print("having label ", len(retina_boxes))

#     for box, score, label in zip(retina_boxes, retina_prob, retina_label):
#         #print("score is ",score.item(), " against ", CONS_TRHESHOLD, "label is ", label.item())
#         if(score.item() > CONS_TRHESHOLD and label.item() in CATEGORY_INFO_OBJECTIVE.keys()):
#             masks, scores, logits = sam_model.predict(
#                 point_coords=None,
#                 point_labels=None,
#                 box= np.array(box.tolist())[None, :] ,
#                 multimask_output=False
#             )

#             if np.any(masks):

#                 mask_values = np.where(masks[0], scores, 0)
#                 final_mask = np.where(mask_values > current_scores , label.item(), final_mask)
#                 current_scores = np.maximum(mask_values, current_scores)


#     lookup = np.vectorize(lambda k: DICT_CLASS_INDEX[k])
#     mask_by_index = lookup(final_mask)

#     return mask_by_index, current_scores



# def process_inference_retina_sam(image,retina_model,sam_model,DICT_CLASS_INDEX ):

#     retina_boxes, retina_prob, retina_label = retina_inference(image,retina_model)

#     mask, score = sam_segmentation_object_detection_retina(image,retina_boxes, retina_prob, retina_label, sam_model, DICT_CLASS_INDEX )

#     score = np.where(mask == 0, 1, score)

#     score_preprocessed  = np.zeros((score.shape[0], score.shape[1], len(DICT_CLASS_INDEX.keys())))

#     for _, val in DICT_CLASS_INDEX.items():
#         score_preprocessed[:,:,val] = np.where(mask == val, score, 0 )

#     score_preprocessed[:,:,0] = 1-np.sum(score_preprocessed[:,:,1:],axis=-1)

#     return mask,  score_preprocessed


# def process_inference_corrector_block(image,result_prev_block,unet_final_model):
#     # Preprocesmiento para que encaje con el modelo (entrenado con shape 256x256 comun a todas las imagenes)
#     target_size = (256, 256)
#     img_resized = cv2.resize(image, target_size) 
#     img_resized = img_resized.astype(np.float32) / 255.0
  

#     pipeline_output =  result_prev_block.astype(np.uint8)
#     pipeline_output = cv2.resize(pipeline_output, target_size)
#     pipeline_output = tf.expand_dims(pipeline_output, axis=-1) 
#     #pipeline_output  = pipeline_output.tobytes()
#     final_input = tf.concat([img_resized, tf.cast(pipeline_output, tf.float32)], axis=-1)


#     # Se añade la dimension del batch para poder aplicar la inferencia
#     input_tensor = np.expand_dims(final_input,axis=0)

#     # Ejecucion de la inferencia
#     prediction = unet_final_model.predict(input_tensor)  

#     # Postprocesamiento
#     pred_mask = np.argmax(prediction[0],axis=-1) 
#     prediction  = prediction[0]


#     return pred_mask,prediction 


# def process_inference_segmentator_block(image,yolo_model,sam_model,DICT_CLASS_INDEX):
#     yolo_boxes, yolo_prob, yolo_label = yolov8_inference(image,yolo_model)
#     mask, score = sam_segmentation_object_detection_yolo(image,yolo_boxes, yolo_prob, yolo_label, sam_model, DICT_CLASS_INDEX )

#     return mask, score


# def process_inference_full_pipeline(image,unet_final_model,yolo_model,sam_model, DICT_CLASS_INDEX = DICT_CLASS_INDEX ):

        
#         result, _  = process_inference_segmentator_block(image,yolo_model,sam_model,DICT_CLASS_INDEX)
#         pred_mask,prediction = process_inference_corrector_block(image,result,unet_final_model)

#         return pred_mask,prediction




# TEXT_PROMPTS = ["a photo of a " + obj + " surrounded by dark" for obj in OBJECTIVES] +  ["a photo of a something surrounded by dark" , "a photo of a background surrounded by dark", "A clear blue sky above the horizon, with silhouettes of buildings and people below"]



# def prepare_clip_input_from_mask(img, mask, clip_proprocess ):
#     coords = np.argwhere(mask)
#     if coords.size == 0:
#         return None

#     y0, x0 = coords.min(axis=0)
#     y1, x1 = coords.max(axis=0) + 1

#     cropped_img = img[y0:y1, x0:x1]
#     cropped_mask = mask[y0:y1, x0:x1]

#     cropped_img = np.copy(cropped_img)
#     cropped_img[~cropped_mask] = 0

#     pil_image = Image.fromarray(cropped_img)

#     clip_input = clip_proprocess(pil_image).unsqueeze(0).to("cpu")
#     return clip_input, pil_image



# def inference_sam_clip(image, clip_model,clip_preprocess,sam_model,text_prompts = TEXT_PROMPTS):

#     result_sam = sam_model.generate(image)
#     result = np.zeros((image.shape[0], image.shape[1]))

#     text_tokens = clip.tokenize(text_prompts).to('cpu')


#     for i, mask_dict in enumerate(result_sam):
#         mask = mask_dict["segmentation"]  
#         clip_input, cropped_pil = prepare_clip_input_from_mask(image, mask, clip_proprocess = clip_preprocess)

#         if clip_input is None:
#             continue

#         # Encode image and text
#         with torch.no_grad():
#             image_features = clip_model.encode_image(clip_input)
#             text_features = clip_model.encode_text(text_tokens)

#             # Normalize
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_features /= text_features.norm(dim=-1, keepdim=True)

#             # Cosine similarity
#             similarity = (image_features @ text_features.T).squeeze(0)
#             best_idx = similarity.argmax().item()

#             if(best_idx<10):
#                 result[mask] = best_idx + 1

#     return result


# def preprocess_model_output(mask):
#     mask = mask.astype(int)
#     one_hot = np.eye(11)[mask]
#     one_hot = one_hot.transpose(2, 0, 1)
#     return one_hot


# def process_inference_sam_clip(image,clip_model,clip_preprocess,sam_model_segment_everything,text_prompts = TEXT_PROMPTS ):

        
#         result = inference_sam_clip(image, clip_model,clip_preprocess,sam_model_segment_everything,text_prompts = text_prompts)
#         proprocessed_mask = preprocess_model_output(result)
#         proprocessed_mask = np.transpose(proprocessed_mask, (1, 2, 0))

#         return result, proprocessed_mask


# ############################################
# #############  MODEL LOADING  ##############
# ############################################

# model_unet_baseline = load_model(BASE_MODEL_UNET, compile=False)
# yolo_model = YOLO("yolov8s.pt") 
# clip_model, clip_preprocess = clip.load("ViT-B/32", device='cpu')

# model_type = "vit_b"
# sam = sam_model_registry[model_type](checkpoint=DIR_SAM)
# sam.to(device= "cpu" )
# sam_model = SamPredictor(sam)

# retina_model = retinanet_resnet50_fpn(pretrained=True)
# retina_model.eval()  

# model_final_pipeline_unet = keras.models.load_model(FINAL_MODEL_UNET, compile=False)

# sam_model_segment_everything = SamAutomaticMaskGenerator(sam)

# ###########################################

# ADDITIONAL_PARAMS = {
#     "model_unet_baseline": model_unet_baseline,
#     "clip_model": clip_model,
#     "clip_preprocess":clip_preprocess,
#     "sam_model_segment_everything": sam_model_segment_everything, 
#     "yolo_model": yolo_model,
#     "sam_model": sam_model,
#     "dict_class_index": DICT_CLASS_INDEX,
#     "retina_model": retina_model,
#     "model_final_pipeline_unet": model_final_pipeline_unet
# }

from typing import Literal
def inference_model_pipeline(image,
                             type_model = Literal["unet_base", "sam_clip", "yolo_sam", "retinanet_sam", "final_model"] ,
                             **kwargs ):
    return
    
#     for key, value in ADDITIONAL_PARAMS.items():
#         kwargs.setdefault(key, value)
    
#     if(type_model == "unet_base" ):
#         model_unet_baseline = kwargs.get("model_unet_baseline", None)
#         pred_mask, probs = process_inference_base_unet(image, model_unet_baseline)

#     elif(type_model == "sam_clip"):
#          clip_model = kwargs.get("clip_model", None)
#          clip_preprocess = kwargs.get("clip_preprocess", None)
#          sam_model_segment_everything = kwargs.get("sam_model_segment_everything", None)
#          pred_mask, probs = process_inference_sam_clip(image,clip_model,clip_preprocess,sam_model_segment_everything )
    
#     elif(type_model == "yolo_sam" ):
#         yolo_model = kwargs.get("yolo_model", None)
#         sam_model = kwargs.get("sam_model", None)
#         DICT_CLASS_INDEX = kwargs.get("DICT_CLASS_INDEX", None)
#         pred_mask, probs = process_inference_yolo_sam(image,yolo_model,sam_model,DICT_CLASS_INDEX )

#     elif(type_model == "retinanet_sam" ):
#         retina_model = kwargs.get("retina_model", None)
#         sam_model = kwargs.get("sam_model", None)
#         DICT_CLASS_INDEX = kwargs.get("DICT_CLASS_INDEX", None)
#         pred_mask, probs =  process_inference_retina_sam(image,retina_model,sam_model,DICT_CLASS_INDEX )  

#     elif(type_model == "final_model" ):
#         model_final_pipeline_unet = kwargs.get("model_final_pipeline_unet", None)
#         yolo_model = kwargs.get("yolo_model", None)
#         sam_model = kwargs.get("sam_model", None)
#         DICT_CLASS_INDEX = kwargs.get("DICT_CLASS_INDEX", None)
#         pred_mask, probs = process_inference_full_pipeline(image,model_final_pipeline_unet,yolo_model,sam_model, DICT_CLASS_INDEX = DICT_CLASS_INDEX )

#     else:
#         return None
    

#     return pred_mask, probs 

