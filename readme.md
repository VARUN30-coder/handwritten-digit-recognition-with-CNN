#  Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits (0–9) using a CNN trained on the MNIST dataset, served via a modern Streamlit web app.


Dataset
This project uses the MNIST dataset containing:

60,000 training images
10,000 testing images
28x28 grayscale handwritten digits

---

##  Project Structure

```
handwritten-digit-recognition/
├── app.py
├── train_model.py
├── digit_model.h5
├── requirements.txt
├── README.md
└── assets/
    ├── accuracy.png
    ├── loss.png
    └── confusion.png
```

---

##  Technologies Used

- Python 3.11
- TensorFlow / Keras
- OpenCV
- Pillow
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Streamlit

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/handwritten-digit-recognition.git
cd handwritten-digit-recognition

# 2. Create virtual environment (Python 3.11)
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

##  How to Run

```bash
# Step 1 — Train the model
python train_model.py

# Step 2 — Launch the app
streamlit run app.py
```

Open browser at → **http://localhost:8501**

---

##  Features

- **Upload Tab** — Upload any digit image, get instant prediction
- **Draw Tab** — Draw a digit on canvas, click Predict
- **Confidence Score** — Shows how sure the model is
- **Probability Chart** — Shows scores for all 10 digits
- **Confusion Matrix** — Visual evaluation of model performance

---

##  CNN Architecture

```
Input (28×28×1)
→ Conv2D(32) → MaxPool
→ Conv2D(64) → MaxPool
→ Conv2D(64)
→ Flatten
→ Dense(128) + Dropout(0.3)
→ Dense(10, Softmax)
```

---

##  Results

| Metric | Value |
|---|---|
| Test Accuracy | ~99.2% |
| Test Loss | ~0.025 |
| Training Time | ~3–5 min (CPU) |

---

---

##  Future Improvements

- TensorFlow Lite conversion for faster inference
- REST API using FastAPI
- Multi-digit recognition
- Deploy on Streamlit Cloud / Hugging Face Spaces

---

##  License

MIT License — free to use and modify.
