"""
debug_launch.py  —  Run this to find EXACTLY where app.py crashes
Usage:
    python debug_launch.py
"""
import traceback
import sys

print("Step 1: Testing basic imports...")
try:
    import sqlite3; print("  [OK] sqlite3")
    from flask import Flask, render_template, Response, request, jsonify; print("  [OK] flask")
    import cv2; print("  [OK] cv2")
    import numpy as np; print("  [OK] numpy")
    import mediapipe as mp; print("  [OK] mediapipe")
    import pickle; print("  [OK] pickle")
    import pandas as pd; print("  [OK] pandas")
    import time; print("  [OK] time")
    import nltk; print("  [OK] nltk")
    import os; print("  [OK] os")
    import threading; print("  [OK] threading")
except Exception as e:
    print(f"\n  CRASH during imports: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 2: Testing speech_recognition + microphone...")
try:
    import speech_recognition as sr
    print("  [OK] speech_recognition imported")
    mic = sr.Microphone()
    print("  [OK] sr.Microphone() created")
except Exception as e:
    print(f"  [WARNING] Microphone issue: {e}")
    print("  (This won't stop the app — speech input just won't work)")

print("\nStep 3: Testing googletrans...")
try:
    from googletrans import Translator
    print("  [OK] googletrans imported")
except Exception as e:
    print(f"  [CRASH] googletrans: {e}")
    traceback.print_exc()

print("\nStep 4: Testing gTTS...")
try:
    from gtts import gTTS
    print("  [OK] gTTS imported")
except Exception as e:
    print(f"  [CRASH] gTTS: {e}")
    traceback.print_exc()

print("\nStep 5: Testing playsound...")
try:
    from playsound import playsound
    print("  [OK] playsound imported")
except Exception as e:
    print(f"  [CRASH] playsound: {e}")
    traceback.print_exc()

print("\nStep 6: Testing mediapipe holistic init...")
try:
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    holistic.close()
    print("  [OK] MediaPipe Holistic init OK")
except Exception as e:
    print(f"  [CRASH] MediaPipe Holistic: {e}")
    traceback.print_exc()

print("\nStep 7: Loading hand_sign.pkl model...")
try:
    import os
    if not os.path.exists("hand_sign.pkl"):
        print("  [CRASH] hand_sign.pkl NOT FOUND in current folder!")
        print(f"  Current directory: {os.getcwd()}")
        print(f"  Files here: {os.listdir('.')}")
    else:
        with open("hand_sign.pkl", "rb") as f:
            model = pickle.load(f)
        print(f"  [OK] model loaded: {type(model)}")
except Exception as e:
    print(f"  [CRASH] Loading pkl: {e}")
    traceback.print_exc()

print("\nStep 8: Testing NLTK data...")
try:
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet',   quiet=True)
    nltk.download('stopwords', quiet=True)
    wnl = nltk.stem.WordNetLemmatizer()
    tokens = nltk.tokenize.word_tokenize("hello world")
    print(f"  [OK] NLTK tokenize works: {tokens}")
except Exception as e:
    print(f"  [CRASH] NLTK: {e}")
    traceback.print_exc()

print("\nStep 9: Testing Flask app creation...")
try:
    app = Flask(__name__)
    print("  [OK] Flask app created")
except Exception as e:
    print(f"  [CRASH] Flask: {e}")
    traceback.print_exc()

print("\nStep 10: Trying to start Flask server...")
try:
    @app.route('/test')
    def test_route():
        return "OK"
    print("  Starting on http://127.0.0.1:685 ...")
    print("  Open browser and go to http://127.0.0.1:685/test")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=False, port=685, use_reloader=False)
except Exception as e:
    print(f"  [CRASH] Flask run: {e}")
    traceback.print_exc()
