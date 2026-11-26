"""
check_setup.py  —  Run this FIRST before running app.py
It tells you exactly which packages are missing and what to install.

Usage:
    python check_setup.py
"""

import sys
import importlib

print("=" * 60)
print("  SIGN LANGUAGE PROJECT — SETUP CHECKER")
print("=" * 60)
print(f"\nPython version: {sys.version}\n")

REQUIRED = [
    # (import_name,        pip_package_name,          notes)
    ("flask",             "flask",                    ""),
    ("cv2",               "opencv-python",            ""),
    ("numpy",             "numpy",                    ""),
    ("mediapipe",         "mediapipe",                ""),
    ("pandas",            "pandas",                   ""),
    ("sklearn",           "scikit-learn",             ""),
    ("nltk",              "nltk",                     ""),
    ("speech_recognition","SpeechRecognition",        ""),
    ("gtts",              "gTTS",                     ""),
    ("googletrans",       "googletrans==4.0.0rc1",    "must be exactly 4.0.0rc1"),
    ("playsound",         "playsound==1.2.2",         "must be exactly 1.2.2"),
    ("werkzeug",          "werkzeug",                 ""),
]

OPTIONAL = [
    ("pyttsx3",  "pyttsx3",  "not used in debugged app — skip if install fails"),
    ("pydub",    "pydub",    "not used in debugged app — skip if install fails"),
    ("pyaudio",  "pyaudio",  "needed only for microphone/speech input"),
]

missing_required = []
missing_optional = []

print("── REQUIRED PACKAGES ──────────────────────────────────")
for imp, pkg, note in REQUIRED:
    try:
        m = importlib.import_module(imp)
        ver = getattr(m, "__version__", "?")
        print(f"  [OK]  {imp:<22} version={ver}")
    except ImportError:
        print(f"  [MISSING]  {imp:<22}  →  pip install {pkg}  {note}")
        missing_required.append(pkg)

print("\n── OPTIONAL PACKAGES ───────────────────────────────────")
for imp, pkg, note in OPTIONAL:
    try:
        m = importlib.import_module(imp)
        ver = getattr(m, "__version__", "?")
        print(f"  [OK]  {imp:<22} version={ver}")
    except ImportError:
        print(f"  [missing] {imp:<22}  →  pip install {pkg}  ({note})")
        missing_optional.append(pkg)

print("\n── OTHER CHECKS ────────────────────────────────────────")

import os
files = ["hand_sign.pkl", "finals.db", "assets", "templates", "static"]
for f in files:
    exists = os.path.exists(f)
    status = "[OK]  " if exists else "[MISSING]"
    print(f"  {status}  {f}")

print("\n── SUMMARY ─────────────────────────────────────────────")
if missing_required:
    print("\n  ⚠  Run this command to fix missing packages:\n")
    print("  pip install " + " ".join(missing_required))
else:
    print("\n  ✅  All required packages are installed!")
    print("  You can now run:  python app.py")

if missing_optional:
    print("\n  Optional (only if you use microphone):")
    print("  pip install " + " ".join(missing_optional))

print("\n" + "=" * 60)
