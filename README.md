# Sign Language Recognition and Translation System

## About the Project

The Sign Language Recognition and Translation System is a Python-based application designed to recognise hand gestures and convert them into meaningful text and speech. It uses computer vision and machine learning techniques to identify gestures and help improve communication between sign language users and others.

## Features

- **Hand Detection:** Detects hand movements and extracts hand landmarks using MediaPipe.
- **Gesture Recognition:** Uses a Random Forest classifier to recognise predefined hand gestures.
- **Text Conversion:** Converts recognised gestures into corresponding text.
- **Multilingual Speech Output:** Converts recognised text into speech in supported languages.
- **Real-Time Recognition:** Supports gesture recognition through a webcam.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| MediaPipe | Hand landmark detection |
| Random Forest | Gesture classification |
| OpenCV | Webcam and image processing (if used) |
| Text-to-Speech | Speech output |
 
## How It Works

1. The webcam captures hand gestures.
2. MediaPipe identifies hand landmarks.
3. The extracted landmark data is passed to the trained Random Forest model.
4. The model predicts the corresponding gesture.
5. The recognised gesture is converted into text.
6. The text is converted into speech in the supported language.

## System Workflow

Webcam Input → Hand Landmark Detection → Feature Extraction → Random Forest Classification → Text Output → Speech Output

## Installation and Setup

### Prerequisites

- Python 3.x
- Webcam
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/Giri922005/sign-language-recognition-and-translation.git
```

### 2. Navigate to the Project Directory

```bash
cd sign-language-recognition-and-translation
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Run the main Python file:

```bash
python main.py
```

Note: Replace `main.py` with your actual application entry-point filename if it is different.

## Project Structure

```text
sign-language-recognition-and-translation/
│
├── dataset/             # Gesture dataset
├── models/              # Trained machine learning model
├── src/                 # Source code
├── screenshots/         # Application screenshots
├── requirements.txt     # Python dependencies
├── main.py              # Application entry point
└── README.md            # Project documentation
```

The folder names above are a suggested structure. Adjust them to match your actual project.

## Applications

- Assist communication between sign language users and others.
- Convert recognised hand gestures into readable text.
- Provide speech output for easier communication.
- Demonstrate the use of computer vision and machine learning in accessibility applications.

## Future Improvements

- Support more sign language gestures.
- Improve recognition accuracy under different lighting conditions.
- Add additional language options for speech output.
- Develop a user-friendly web or mobile interface.
- Improve continuous gesture recognition.

## Author

**Giridharan G**

---

*Developed as an academic project to explore computer vision, machine learning, and assistive technology.*
