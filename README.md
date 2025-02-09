# TFM



El trabajo se ha estructurado siguiendo la metodología CRISP-DM, organizada de la siguiente manera:

1) **Problem Understanding y Data Understanding (I)**: Se aborda en el siguiente fichero [`src/data_loading.ipynb`](src/data_loading.ipynb).
2) **Data Understanding (II)**: Se desarrolla en el fichero [`src\exploratory_data_analysis.ipynb`](src\exploratory_data_analysis.ipynb).
2) **Data Preparation **: presente en el fichero [`src\feature_extraction.ipynb`](src\feature_extraction.ipynb).
2) **Data Modeling (I)**: primer modelo entrenado, en este caso la U-Net, que servirá como baseline, implementado en : [`src/data_analysis.ipynb`](src/data_analysis.ipynb).
2) **Data Modeling (II)**: segundo modelo, en este caso un ensemble donde se usan modelos fundacionales [`src/data_analysis.ipynb`](src/data_analysis.ipynb).
2) **Model evaluation**: comparativa de resultados de ambos modelos y conclusiones de los mismos[`src/data_analysis.ipynb`](src/data_analysis.ipynb).