# 🌿 Leaf Disease Detection 

This project implements a **deep learning-based leaf disease detection system** using **CNN/ResNet50** architecture.  
It identifies diseases from leaf images using a trained model, preprocessing pipelines, and real-time prediction scripts.

---
🎯 Purpose of the Project

The purpose of this project is to build an automated, accurate, and fast leaf disease detection system using deep learning.
Plant diseases significantly impact crop production, and early detection helps farmers prevent large-scale losses.

This project aims to:

Identify plant leaf diseases from images

Assist farmers and researchers with AI-powered diagnosis

Reduce manual inspection time

Improve agricultural yield through early intervention

🌟 Advantages

High Accuracy (97.7%) — Reliable predictions for multiple leaf diseases

Automated Detection — No expert knowledge required

Fast & Efficient — Real-time predictions using optimized CNN

Transfer Learning — Works even with smaller datasets

Easy to Use — Just provide an image and get the disease result

Scalable — Can be deployed on web, mobile, or edge devices

💡 Why People Choose This Model

People choose this leaf disease detection system because:

Very high accuracy (97.7%) — nearly expert-level detection

Supports real-time prediction — great for field use

Lightweight & deployable — works on normal systems

Modular code design — easy to extend, modify, or retrain

Supports multiple diseases — not limited to a single crop

Open-source — free to use and customize

This makes it ideal for:

Farmers

Researchers

Students

Agriculture startups

AI/ML learners

📊 Model Performance

Your model reached:

⭐ Overall Accuracy: 97.7%
Metric	Value
Accuracy	97.7%
Precision	High
Recall	High
F1-Score	Excellent
Loss	Low

You can add the exact numbers after running evaluation.

⚙️ Functions of the System

The system includes multiple core functions:

✔ 1. Preprocessing

Image resizing

Normalization

Data augmentation

Noise reduction

✔ 2. Training

Base training (transfer learning)

Fine-tuning (unfreezing deeper layers)

Model saving & checkpointing

✔ 3. Prediction

Single image prediction

Confidence score output

Class label identification

✔ 4. Evaluation

Accuracy & loss calculation

Confusion matrix

Precision, recall, F1-score

✔ 5. Deployment

Web-based interface using app.py

Accepts image uploads and returns prediction instantly
## 🚀 Features

- ✔️ High-accuracy disease classification using **ResNet50**
- ✔️ Training + Fine-tuning scripts included  
- ✔️ Real-time prediction (CLI or app interface)
- ✔️ Preprocessing & evaluation modules
- ✔️ Professional project folder structure  
- ✔️ Ready for deployment (Flask / Streamlit)

---

## 📂 Folder Structure

leaf-disease-detection/
│
├── app/
│ └── app.py # Web app interface (Flask/Streamlit)
│
├── models/
│ └── model_link.txt # Contains Google Drive link to trained model
│
├── src/
│ ├── train_model.py # Initial training script
│ ├── train_phase2.py # Fine-tuning / second phase
│ ├── preprocess.py # Image preprocessing pipeline
│ ├── evaluate_model.py # Evaluation metrics & confusion matrix
│ ├── predict.py # Predict on a single input image
│ ├── predict_leaf.py # Prediction helper script
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## 📦 Download Trained Model

GitHub does not allow large models (>25MB),  
so the trained model is stored on Google Drive.

👉 **Download trained model:**  
https://drive.google.com/drive/folders/1jkyc2dgn_w17BG_A7eIVzoDI7ZKocjoJ?usp=drive_link

After downloading, place it inside:

models/
└── leaf_model.h5

yaml
Copy code

---

## 🧠 Model Architecture

The model is based on **ResNet50** pretrained on ImageNet.

Architecture flow:

Input Image → Preprocessing → ResNet50 (Frozen Layers)
→ GlobalAveragePooling → Dense Layers → Softmax Output

yaml
Copy code

Benefits:
- 👍 Transfer learning → faster and more accurate  
- 👍 Works with smaller datasets  
- 👍 High generalization and robustness  

---

## ⚙️ Installation

First clone the repository:

```bash
git clone https://github.com/AbhayPratap0307/leaf-disease-detection.git
cd leaf-disease-detection
Install dependencies:

bash
Copy code
pip install -r requirements.txt
🏋️ Training the Model
Phase 1 — Base Training
bash
Copy code
python src/train_model.py
Phase 2 — Fine-tuning
bash
Copy code
python src/train_phase2.py
🔍 Making Predictions
Predict on a single image:
bash
Copy code
python src/predict.py --image sample_leaf.jpg
Predict using helper:
bash
Copy code
python src/predict_leaf.py
Output includes:

Predicted disease

Confidence score

📊 Evaluation
Evaluate model accuracy, loss, precision, recall, F1-score:

bash
Copy code
python src/evaluate_model.py
Graphs & confusion matrix will be generated.

📚 Dataset
You may use:

PlantVillage Dataset

Custom datasets (collected leaf images)

If using a large dataset, place it in Google Drive or Kaggle and link it here.

📈 Results (Add yours here)
Metric	Value
Accuracy	95% (example)
Loss	0.12 (example)
F1-Score	0.94

You can update this table after evaluating your model.

💻 Tech Stack
Python

TensorFlow / Keras

NumPy

Pandas

Matplotlib

OpenCV

Flask / Streamlit (optional app)

📦 Deployment (Optional)
To run the app:

bash
Copy code
python app/app.py
You can deploy on:

Streamlit Cloud

Render

HuggingFace Spaces

Heroku

👤 Author
Abhay Pratap Yadav
GitHub: https://github.com/AbhayPratap0307


📝 License
This project is covered under the MIT License.

THANK YOU

---


