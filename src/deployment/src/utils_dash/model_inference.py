import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor


#### CONST
MODELS = {}




###### UTILS INFERENCE ######

def get_model_instance(model:str):
    
    global MODELS

    if model in MODELS:
        return MODELS[model]

    if(model == "retinanet"):
        pass


    elif(model == "sam"):
        sam_checkpoint = r"C:\Users\ruben\Desktop\code_tfm\models\SAM\sam_vit_b_01ec64.pth"# r"C:\Users\ruben\Desktop\code_tfm\models\SAM\sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device= "cpu" )
        sam_model = SamPredictor(sam)
        MODELS["sam"] = yolo_model

        return sam_model



    elif(model == "yolo"):
        yolo_model = YOLO("yolov8s.pt")
        yolo_model.overrides['verbose'] = False

        MODELS["yolo"] = yolo_model

        return yolo_model
    

    print("Model unavailable provided")
    return None



### YOLO + SAM INFERENCE

def inference_yolo_sam(image, threshold, categories_index_by_name, category_info_objetive):

    yolo_model = get_model_instance("yolo")
    sam_model = get_model_instance("sam")

    print("models loaded")

    mask, _ = process_inference_yolo_sam(image, yolo_model, sam_model,threshold, categories_index_by_name, category_info_objetive)


    return mask




def process_inference_yolo_sam(image, yolo_model, sam_model,threshold, categories_index_by_name, category_info_objetive):

    results = yolo_model(image)
    yolo_boxes = []
    yolo_prob = []
    yolo_label = []
    
    # YOLO INFERENCE
    for box_info in results[0]:
        x1,y1,x2,y2, conf, class_id = box_info.boxes.data[0].tolist()
        class_name = yolo_model.names[int(class_id)]
        #print("predicted a ",class_name )

        if (class_name) in categories_index_by_name:
            yolo_boxes.append( (x1,y1,x2,y2))
            yolo_prob.append(conf)
            yolo_label.append(categories_index_by_name[class_name])

    print("yolo inefrence generated")

    inference = {}
    inference["boxes"] = yolo_boxes
    inference["scores"] = yolo_prob
    inference["labels"] = yolo_label


    # SAM INFERENCE

    sam_model.set_image(image)
    final_mask = np.zeros(image.shape[:2], dtype=np.int8)
    current_scores = np.zeros(image.shape[:2], dtype=np.float32)

    masks_image = []
    scores_image = []
    #logits_image = []
    labels_image = []
    #category_info_objetive = {v: k for k, v in categories_index_by_name.items()}

    
    for box, score, label in zip(inference['boxes'], inference['scores'], inference['labels']):
        if(score > threshold and label in category_info_objetive.keys()):
            masks, scores, _ = sam_model.predict(
                point_coords=None,
                point_labels=None,
                box= np.array([round(x) for x in box]),
                multimask_output=False
            )
            if np.any(masks):
                masks_image.append(masks)
                scores_image.append(scores)
                #logits_image.append(logits[0])
                labels_image.append(label)

                mask_values = np.where(masks, scores, 0)
                final_mask = np.where(mask_values > current_scores , label, final_mask)
                current_scores = np.maximum(mask_values, current_scores)

    print("sam inference loaded")
    return final_mask[0], current_scores[0]


### Plotting function
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






    
