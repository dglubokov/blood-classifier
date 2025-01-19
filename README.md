# **Blood-Classifer 🩸**

A machine learning-powered tool for classifying blood and bone marrow cells using microscopy image data.

![Project Concept](concept_tree.png)

---

## **Table of Contents**
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Training](#training)
   - [System Usage](#system-usage)
5. [Development](#development)
6. [Contributing](#contributing)
7. [License](#license)

---

## **Overview**
This project focuses on the classification of blood and bone marrow cells using deep learning techniques. It utilizes data from the **Cancer Imaging Archive** and provides a system for training models, interpreting results, and deploying a user-friendly interface with back-end and front-end support.

Key features:
- Pre-trained models for cell classification.
- Customizable training pipeline for new datasets.
- Dockerized deployment for seamless integration.

---

## **Dataset**
The project uses the following dataset:

> **Matek, C., Krappe, S., Münzenmayer, C., Haferlach, T., & Marr, C. (2021).**  
> *An Expert-Annotated Dataset of Bone Marrow Cytology in Hematologic Malignancies [Data set].*  
> [The Cancer Imaging Archive](https://doi.org/10.7937/TCIA.AXH3-T579)  

This dataset includes expertly annotated images of bone marrow cytology, aiding the development of robust classification models.

---

## **Installation**

### **Prerequisites**
- Python 3.8+
- NVIDIA GPU with CUDA support for model training
- Docker and Docker Compose

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/dglubokov/blood-classifier.git
   cd blood-classifier
   ```

2. Install required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Docker is installed and running:
   ```bash
   docker --version
   docker-compose --version
   ```

---

## **Usage**

### **Training**
1. Prepare your dataset and place it in the appropriate directory.
2. Modify paths in the Jupyter notebook file [./experiments/5_experiment.ipynb](./experiments/5_experiment.ipynb) to point to your dataset.
3. Run the notebook:
   ```bash
   jupyter notebook ./experiments/5_experiment.ipynb
   ```
   > ⚠️ **Note:** A powerful GPU is required to train the models efficiently.

4. After training, the models and interpreted image examples will be saved locally.

---

### **System Usage**
1. Move the trained models to the `./models/` directory (create this directory if it doesn't exist):
   ```bash
   mkdir models
   mv <your_model_files> ./models/
   ```

2. Start the Dockerized system:
   ```bash
   docker-compose up
   ```

   This will launch:
   - **FastAPI Back-End:** Accessible at [http://0.0.0.0:8082/docs/](http://0.0.0.0:8082/docs/)  
   - **React Front-End:** Accessible at [http://0.0.0.0:3000/](http://0.0.0.0:3000/)

---
