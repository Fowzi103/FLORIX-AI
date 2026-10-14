# 🌿 FLORIX AI — Intelligent Plant Disease Detection

<p align="center">
  <b>AI-powered plant disease recognition using Deep Learning</b><br>
  Helping users explore plant health through leaf image analysis.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN%20%7C%20LSTM-8A2BE2?style=for-the-badge" alt="Deep Learning">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Interface-Gradio%20%7C%20Streamlit-FF4B4B?style=for-the-badge" alt="Interface">
</p>

## 📌 Overview

**FLORIX AI** is a plant disease prediction project that applies deep learning and computer vision to analyze uploaded leaf images and predict a possible disease category.

The project aims to make plant health analysis more accessible through an easy-to-use interface, combining image preprocessing, model inference, and understandable prediction results.

## ✨ Key Features

- 🌱 **Leaf Image Analysis** — Upload a plant leaf image for analysis.
- 🧠 **Deep Learning Prediction** — Use a trained model to classify the input image.
- 📊 **Prediction Confidence** — Display the model's confidence score when available.
- 🖥️ **Interactive Interface** — Provide a user-friendly experience using Gradio or Streamlit.
- ⚡ **Image Preprocessing** — Resize and normalize images to match the model's input requirements.
- 🔍 **Expandable Architecture** — Designed to support additional plant species, disease classes, and model improvements.

## ⚙️ How It Works

1. **Upload:** The user selects a plant leaf image.
2. **Preprocess:** The image is converted to RGB, resized to the model's expected dimensions, and transformed into a numerical array.
3. **Predict:** The trained deep learning model produces class scores.
4. **Interpret:** The application selects the predicted class and calculates its confidence score.
5. **Display:** The interface presents the prediction to the user.

## 🧠 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model development and inference |
| NumPy | Image array processing |
| Pillow (PIL) | Image loading and resizing |
| Gradio / Streamlit | Interactive web interface |
| CNN / other trained architecture | Image classification |

*Update the architecture and interface badges to match the model and framework actually used in your repository.*

## 📁 Project Structure

```text
FLORIX-AI/
├── app.py                 # Application entry point (if applicable)
├── model/                 # Trained model files (if included)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── assets/                # Screenshots and other assets (optional)
```

The structure above is a suggested layout. Adjust it to match your actual repository files.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a version compatible with your dependencies
- pip
- The trained model file required by the application
- The dependencies listed in `requirements.txt`

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Fowzi103/FLORIX-AI.git
cd FLORIX-AI
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

For a Gradio application:

```bash
python app.py
```

For a Streamlit application:

```bash
streamlit run app.py
```

Use the command that matches your actual application framework and entry-point filename.

## 📷 Application Preview

Add screenshots of your application here to help visitors understand the interface.

```markdown
![FLORIX AI Home Screen](assets/home.png)
![FLORIX AI Prediction Result](assets/prediction.png)
```

Create an `assets` folder and upload your screenshots before enabling these image links.

## 🌱 Potential Future Enhancements

- Support more plant species and disease categories.
- Compare multiple trained model architectures.
- Display class probabilities and prediction confidence.
- Add a prediction history dashboard.
- Provide disease-care information from reliable agricultural sources.
- Deploy the application for public access.

## ⚠️ Disclaimer

FLORIX AI is an educational machine learning project. Predictions may be incorrect and should not be treated as a definitive plant diagnosis. For important crop-health decisions, consult a qualified agricultural expert.

## 👩‍💻 Author

**Fowziya**

- GitHub: [@Fowzi103](https://github.com/Fowzi103)
- Project Repository: [FLORIX AI](https://github.com/Fowzi103/FLORIX-AI)

---

<p align="center">
  🌿 <b>Building smarter solutions for healthier plants.</b>
</p>
