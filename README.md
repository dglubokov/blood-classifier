# blood-classifier 🩸

Blood and Bone Marrow classifier

![concept](concept_tree.png)

## Used Data

*Matek, C., Krappe, S., Münzenmayer, C., Haferlach, T., & Marr, C. (2021). An Expert-Annotated Dataset of Bone Marrow 
Cytology in Hematologic Malignancies [Data set]. The Cancer Imaging Archive.*
[https://doi.org/10.7937/TCIA.AXH3-T579](https://doi.org/10.7937/TCIA.AXH3-T579)


## Usage

### Training

1. Prepare your data and add it to specific dir.
2. After this do some modifications in [./experiments/5_experiment.ipynb](./experiments/5_experiment.ipynb) where you can add your path to data
3. Run all JupyterNotebook cells (Attention! You need a good GPU to train all models)
4. You will get training models and interpreted image examples

### System usage

1. Move your trained models to [./models/](./models/) dir (you should create it!).
2. Run ```docker-compose up ```

This command runs back-end and front-end containers on specific ports:
- FastAPI back-end endpoints on [http://0.0.0.0:8082/docs/](http://0.0.0.0:8082/docs/)
- React front-end on [http://0.0.0.0:3000](http://0.0.0.0:3000)

### Developing

All code for back-end and front-end are available in appropriate directories.
