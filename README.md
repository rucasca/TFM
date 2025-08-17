# Segmentación semántica del dataset COCO mediante UNET y arquitecturas ensemble 


A lo largo de este proyecto se aborda mediante diferentes notebooks el proceso completo de la resolución de un problema de **segmentación semántica** 🖼️ mediante diferentes **arquitecturas** tanto **convolucionales** como con arquitecturas que usan **mecanismos de atención**, además de su despliegue para la productivización mediante una aplicación web 🌐.
El trabajo se ha estructurado siguiendo la metodología `CRISP-DM`, organizada de la siguiente manera:

### Comprensión del negocio (*Business Understanding* en CRISP_DM) y comprensión de los datos (*Data Understanding*) 🧠📊 
En estas dos fases se incluye tanto el entendimeinto del objetivo del proyecto como una primera carga y evaluación del conjunto de datos, donde se detectan patrones que influirán en fases posteriores de la metodología con el objetivo de la obtención de los mejores resultados posibles que se adecuen de forma óptima al problema definido.

- La carga inicial y el entendimiento del objetivo se desarrollan en el notebook  [`src/01_data_loading.ipynb`](src/01_data_loading.ipynb).
- El análisis exploratorio del conjunto de datos tiene lugar en el notebook [`src/02_exploratory_data_analysis.ipynb`](src/02_exploratory_data_analysis.ipynb).


### Preparación de los datos (*Data Preparation*) 🛠️  
A lo largo de esta fase se incluye la fase de comprensión de las imágenes en dimensiones comunes, la carga del dataset en un formato más eficiente para el entrenamiento de los modelos, en este caso `tf.tfrecord`, y el sampleamiento del conjunto de datos para disminuir el desvalanceo. Además, se definirá la fase de **data augmentation**, que permite la obtención de diferentes muestras a partir del conjunto de datos sampleado que forzarán al modelo a aprender.

El contenido de esta fase se desarrolla en el notebook [`src/03_data_preprocessing.ipynb`](src/03_data_preprocessing.ipynb).

### Modelado de los Datos (*Modeling*) 🤖 
Fase que comprende el entrenamiento de modelos que permitirán resolver el problema definido. En este caso los modelos y arquitecturas empleadas han sido:
1) Modelo baseline, en este caso la UNET 🧬. Contenido en el notebook [`src/04_01_data_modeling_UNET.ipynb`](src/04_01_data_modeling_UNET.ipynb).
2) Arquitectura ensemble empleando YoloV8 (no fundacional, pero entrenado en el mismo conjunto de datos) + SAM 🧪 . Desarrollada en el notebook [`src/04_02_data_modeling_YOLO_SAM.ipynb`](src/04_02_data_modeling_YOLO_SAM.ipynb).
3) Arquitectura ensemble fundacional, con SAM + Retinanet 🧠 . Implementado en [`src/04_03_data_modeling_RetinaNet_SAM.ipynb`](src/04_03_data_modeling_RetinaNet_SAM.ipynb).
4) Arquitectura ensemble inversa 🔄 , donde primero se segmenta y luego se clasifica, usando SAM + CLIP. Contenida en el notebook [`src/04_04_data_modeling_SAM_CLIP.ipynb`](src/04_04_data_modeling_SAM_CLIP.ipynb).
5) Arquitectura ensemble final, donde primero se segmenta y luego se clasifica, usando SAM + CLIP. Contenida en el notebook [`src\04_05_data_modeling_final_model.ipynb`](src\04_05_data_modeling_final_model.ipynb).


### Evaluación de los resultados (*Evaluation*) 📈 

Se comparan los resultados obtenidos por cada uno de los modelos desarrollados atendiendo a diferentes criterios 📊 . 
Esta comparativa de resultados tiene lugar en el fichero [`src/05_results_comparative.ipynb`](src/results_comparative.ipynb).



### Implementación y productivización (*Deployment*)🚀 
Fase que comprende la puesta en funcionamiento de pipelines que permiten el uso de los modelos en un entorno usable en la vida real. En este caso se ha productivizado el modelo mediante una aplicación web desarrollada en el framework `Dash` 💻 .

Esta puede ser encontrada en el directorio [`src/06_deployment/src/app.py`](src/06_deployment/src/app.py) y al inicializarla despliega en un puerto local un aplicativo web que permite el uso de los pipeline de las arquitecturas *ensemble* implementadas de forma intiutiva.




Para la reproducción de los expermientos realizados se facilita un fichero `pyproject.toml` que contiene todas las librerias empleadas y sus versiones correspondientes.
Para su instalación, se han de seguir los pasos siguientes: 
1) Instalación de poetry mediante el *pip*
2) Ejecución del comando por consola `poetry install`
3) Ejecución de los notebook. En el caso de que VSCode no seleccione el entorno creado de forma automática, selección manual del mismo mediante `CTRL + SHIT + P` -> Select Python interpreter
4) Rellenado de las variables del entorno del fichero `.env` con las direcciones de memoria en locar que apunten a los pesos de los modelos fundacionales descargados desde sus respectivos enlaces.
5) Para el despliegie de la web, ejecutar el fichero [`src/deployment/src/app.py`](src/deployment/src/app.py)



