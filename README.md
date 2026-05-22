# 🎓 Student Emotion Detection System

A real-time facial emotion recognition system designed to help teachers monitor students' emotional engagement in classrooms or online learning environments.

---

## 📸 Features

- **Live webcam feed** with face detection overlay
- **Real-time emotion classification**: Happy, Sad, Angry, Confused, Bored, Surprised, Neutral
- **Confidence scores** and per-emotion breakdown
- **Session summary** with percentage bars
- **Recent log** of detected emotions
- **Export log** as JSON file
- **Simple, dark-themed UI** for teachers and students

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| Emotion Model | FER+ ONNX (OpenCV DNN) |
| Face Detection | OpenCV Haar Cascade |
| Frontend | HTML, CSS, JavaScript |
| Data Storage | JSON |

---

## 📦 Requirements

- Python 3.9 – 3.14
- Webcam / camera

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/BertoBerlo12345/Emotion-Detection-System.git
cd Emotion-Detection-System
```

### 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

> The emotion model (`emotion-ferplus-8.onnx`) will be **downloaded automatically** on first run (~33 MB).

---

## 📁 Project Structure

```
Emotion Detection System/
├── app.py                  # Flask backend + emotion detection logic
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── templates/
│   └── index.html          # Main UI page
├── static/
│   ├── css/
│   │   └── style.css       # Dark theme stylesheet
│   └── js/
│       └── main.js         # Real-time polling & UI updates
└── model/                  # Auto-created on first run
    └── emotion-ferplus-8.onnx
```

---

## 🎭 Emotion Labels

| Model Output | Classroom Label |
|-------------|----------------|
| Happy | Happy 😄 |
| Sad | Sad 😢 |
| Angry | Angry 😠 |
| Scared | Confused 😕 |
| Disgusted | Bored 😑 |
| Surprised | Surprised 😲 |
| Neutral | Neutral 😐 |

---

## 📊 Dataset

The emotion model was trained on the **FER-2013** dataset from Kaggle:
- 35,887 grayscale face images (48×48 px)
- 7 emotion categories
- Source: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 👥 Group Members

- *(Add your group members here)*

---

## 📄 License

For academic use only.
