# 🌿 Leaf Disease Detection 

This project implements a **deep learning-based leaf disease detection system** using **CNN/ResNet50** architecture.  
It identifies diseases from leaf images using a trained model, preprocessing pipelines, and real-time prediction scripts.

---

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

