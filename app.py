# ============================================================
#  SMART SIGN LEARNING SYSTEM 
# ============================================================
#
#  FIXES IN THIS VERSION:
#  1. Detects BOTH hands (left + right) — works regardless of which hand you show
#  2. Buffer correctly stores full predicted word (not just first token)
#  3. whole_word_text built with "".join so words appear without extra spaces
#  4. Language TTS works for all 5 languages via /play/<lang>
#  5. Thread-safe state with Lock
#  6. generate_frames runs in background thread so browser stays live
# ============================================================

import sqlite3
import os
import time
import pickle
import threading

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import nltk

from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename

from googletrans import Translator
from gtts import gTTS
from playsound import playsound

import speech_recognition as sr

# ── NLTK downloads ──────────────────────────────────────────────────────────
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('wordnet',    quiet=True)
nltk.download('stopwords',  quiet=True)

porter = nltk.stem.PorterStemmer()
wnl    = nltk.stem.WordNetLemmatizer()

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE = "finals.db"

def create_table():
    conn = sqlite3.connect(DATABASE)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS register "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, mail TEXT, password TEXT)"
    )
    conn.commit()
    conn.close()

create_table()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ── Pre-trained model ─────────────────────────────────────────────────────────
MODEL_PATH = "hand_sign.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"'{MODEL_PATH}' not found. Run training.py first."
    )
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ── Shared state ──────────────────────────────────────────────────────────────
state_lock           = threading.Lock()
detected_letters     = []          # list of captured words/letters
whole_word_text      = ""          # full sentence shown in browser
last_detected_letter = ""          # current live prediction shown in OpenCV window
camera_running       = False       # flag so only one camera thread runs at a time

# ── Language codes ────────────────────────────────────────────────────────────
LANG_CODES = {
    "english":   "en",
    "tamil":     "ta",
    "hindi":     "hi",
    "malayalam": "ml",
    "kannada":   "kn",
}

# ── Popup helper ─────────────────────────────────────────────────────────────
def display_popup(msg):
    return f'<script>alert("{msg}"); window.history.back();</script>'


# ════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ════════════════════════════════════════════════════════════

@app.route('/', methods=["GET", "POST"])
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name         = request.form['name']
        mail         = request.form['mail']
        password     = request.form['password']
        confirm_pass = request.form['confirm_pass']

        if password != confirm_pass:
            return display_popup("Passwords do not match.")

        conn = sqlite3.connect(DATABASE)
        cur  = conn.cursor()
        cur.execute("SELECT mail FROM register WHERE mail=?", (mail,))
        if cur.fetchone():
            conn.close()
            return display_popup("Email already registered.")

        cur.execute(
            "INSERT INTO register (name, mail, password) VALUES (?,?,?)",
            (name, mail, password)
        )
        conn.commit()
        conn.close()
        return render_template('login.html')

    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        mail     = request.form['mail']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        cur  = conn.cursor()
        cur.execute(
            "SELECT * FROM register WHERE mail=? AND password=?",
            (mail, password)
        )
        data = cur.fetchone()
        conn.close()

        if data is None:
            return display_popup("Invalid email or password.")

        return render_template('index.html', title='Smart Sign Learning')

    return render_template('login.html')


# ════════════════════════════════════════════════════════════
#  MAIN PAGES
# ════════════════════════════════════════════════════════════

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/home', methods=["GET", "POST"])
def home():
    return render_template('home.html')

@app.route('/sign', methods=["GET", "POST"])
def sign():
    return render_template('sign.html')


# ════════════════════════════════════════════════════════════
#  SIGN → TEXT  (webcam recognition)
# ════════════════════════════════════════════════════════════

def _get_hand_landmarks(results):
    """
    FIX 1: Try BOTH hands so it works regardless of which hand the user shows.
    The model was trained on left_hand_landmarks, but MediaPipe mirrors
    left/right based on camera orientation. We try left first, then right.
    Returns landmark list or None.
    """
    if results.left_hand_landmarks:
        return results.left_hand_landmarks.landmark
    if results.right_hand_landmarks:
        return results.right_hand_landmarks.landmark
    return None


def run_camera():
    """
    Runs in a background thread. Opens webcam, detects signs.

    Keyboard shortcuts (OpenCV window must be in focus):
      i  → capture current detected word into the sentence buffer
      c  → add a space between words
      p  → speak the buffered sentence aloud (English TTS)
      q  → quit camera
    """
    global detected_letters, last_detected_letter, whole_word_text, camera_running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        camera_running = False
        return

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── MediaPipe processing ──────────────────────────────────────
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # ── Draw landmarks for both hands ─────────────────────────────
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,  22,  10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,  44, 121), thickness=2, circle_radius=2),
            )
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22,  76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
            )

            # ── Prediction (FIX 1: both hands) ───────────────────────────
            body_language_class = "None"
            landmarks = _get_hand_landmarks(results)

            if landmarks:
                pose_row = list(
                    np.array([
                        [lm.x, lm.y, lm.z, lm.visibility]
                        for lm in landmarks
                    ]).flatten()
                )
                X = pd.DataFrame([pose_row])
                try:
                    body_language_class = model.predict(X)[0]
                except Exception as e:
                    print(f"Prediction error: {e}")
                    body_language_class = "Error"

            # Update live letter shown in overlay
            with state_lock:
                last_detected_letter = body_language_class

            # ── Keyboard shortcuts ────────────────────────────────────────
            key = cv2.waitKey(5) & 0xFF

            if key == ord('i'):                        # capture word
                if body_language_class not in ("None", "Error"):
                    with state_lock:
                        detected_letters.append(body_language_class)
                        # FIX 2: join with space between words
                        whole_word_text = " ".join(detected_letters)
                    print(f"Captured: {body_language_class} | Buffer: {whole_word_text}")

            elif key == ord('c'):                      # insert space
                with state_lock:
                    detected_letters.append("")        # empty string → space between words
                    whole_word_text = " ".join(detected_letters).strip()
                print("Space added")

            elif key == ord('p'):                      # speak English
                with state_lock:
                    text_to_speak = whole_word_text
                if text_to_speak.strip():
                    try:
                        tts = gTTS(text=text_to_speak, lang="en")
                        tts.save("speak_out.mp3")
                        playsound("speak_out.mp3")
                        os.remove("speak_out.mp3")
                    except Exception as e:
                        print(f"TTS error: {e}")

            elif key == ord('q'):                      # quit
                break

            # ── Overlay text ──────────────────────────────────────────────
            with state_lock:
                disp_buffer = whole_word_text
                disp_letter = last_detected_letter

            # Background bar for readability
            cv2.rectangle(image, (0, 0), (640, 75), (245, 117, 16), -1)
            cv2.putText(image, "CLASS",
                        (95, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(disp_letter),
                        (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, f"Buffer : {disp_buffer}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "i=capture  c=space  p=speak  q=quit",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Detect Sign  |  i=capture  c=space  p=speak  q=quit", image)

    cap.release()
    cv2.destroyAllWindows()
    camera_running = False
    print("Camera closed.")


@app.route('/generate_frames', methods=["GET", "POST"])
def generate_frames():
    global camera_running
    if not camera_running:
        camera_running = True
        t = threading.Thread(target=run_camera, daemon=True)
        t.start()
    return render_template("home.html")


# ── Live JSON for browser polling ────────────────────────────────────────────
@app.route('/get_data')
def get_data():
    with state_lock:
        return jsonify({'data': whole_word_text})


# ── Clear buffer ──────────────────────────────────────────────────────────────
@app.route('/delete', methods=["GET", "POST"])
def delete():
    global detected_letters, whole_word_text, last_detected_letter
    with state_lock:
        detected_letters.clear()
        whole_word_text      = ""
        last_detected_letter = ""
    print("Buffer cleared.")
    return render_template("home.html")


# ════════════════════════════════════════════════════════════
#  TEXT / SPEECH → SIGN  (Learning module)
# ════════════════════════════════════════════════════════════

ASSETS_LIST = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','after','again','against','age','all','alone','also','and','ask','at',
    'b','be','beautiful','before','best','better','busy','but','bye',
    'c','can','cannot','change','college','come','computer',
    'd','day','distance','do not','do','does not',
    'e','eat','engineer',
    'f','fight','finish','from',
    'g','glitter','go','god','gold','good','great',
    'h','hand','hands','happy','hello','help','her','here','his','home',
    'homepage','how',
    'i','invent','it',
    'j','k','keep','l','language','laugh','learn',
    'm','me','more','my',
    'n','name','next','not','now',
    'o','of','on','our','out',
    'p','pretty','q','r','right',
    's','sad','safe','see','self','sign','sing','so','sound','stay','study',
    't','talk','television','thank you','thank','that','they','this','those',
    'time','to','type',
    'u','us','v','w','walk','wash','way','we','welcome',
    'what','when','where','which','who','whole','whose','why','will',
    'with','without','words','work','world','wrong',
    'x','y','you','your','yourself','z',
]

STOP_WORDS_EXTRA = [
    '@','#','http',':','is','the','are','am','a','it',
    'was','were','an',',','.','?','!',';','/',
]

def text_to_tokens(input_text: str):
    stop = nltk.corpus.stopwords.words('english') + STOP_WORDS_EXTRA
    tokenized = nltk.tokenize.word_tokenize(input_text)
    lemmed    = [wnl.lemmatize(w) for w in tokenized]

    processed = []
    for word in lemmed:
        w = word.lower()
        if w == 'i':
            processed.append('me')
        elif w not in stop:
            processed.append(w)

    tokens = []
    for word in processed:
        if word in ASSETS_LIST:
            tokens.append(f"assets/{word}.mp4")
        else:
            for ch in word:
                if ch in ASSETS_LIST:
                    tokens.append(f"assets/{ch}.mp4")
    return tokens


def stream_sign_videos(input_text: str):
    tokens = text_to_tokens(input_text)
    for asset_path in tokens:
        if not os.path.exists(asset_path):
            print(f"Asset not found, skipping: {asset_path}")
            continue
        label = asset_path.replace("assets/", "").replace(".mp4", "")
        cap   = cv2.VideoCapture(asset_path)
        fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                jpeg.tobytes() +
                b'\r\n\r\n'
            )
            time.sleep(1 / fps)
        cap.release()


@app.route('/video_feed_endpoint', methods=["POST"])
def video_feed_endpoint():
    submitted_text = request.form.get('text', '')
    return Response(
        stream_sign_videos(submitted_text),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ════════════════════════════════════════════════════════════
#  SPEECH INPUT → SIGN
# ════════════════════════════════════════════════════════════

r_recognizer = sr.Recognizer()

@app.route('/speak', methods=["GET", "POST"])
def speak():
    try:
        mic = sr.Microphone()
        with mic as source:
            print("Listening...")
            r_recognizer.adjust_for_ambient_noise(source)
            audio = r_recognizer.listen(source, timeout=5)
        speech_text = r_recognizer.recognize_google(audio).lower()
        print(f"Heard: {speech_text}")
    except Exception as e:
        print(f"Speech error: {e}")
        speech_text = ""

    if not speech_text:
        return render_template('sign.html')

    return Response(
        stream_sign_videos(speech_text),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ════════════════════════════════════════════════════════════
#  MULTI-LANGUAGE TTS  ← FIX 3: all 5 languages work correctly
# ════════════════════════════════════════════════════════════

def generate_speech(text: str, lang_code: str):
    """
    Translates text to target language and plays it via gTTS.
    Works for: en, ta, hi, ml, kn
    """
    print(f"[TTS] Speaking '{text}' → lang='{lang_code}'")
    try:
        # Translate only if not English
        if lang_code != "en":
            translator = Translator()
            translated = translator.translate(text, dest=lang_code).text
            print(f"[TTS] Translated: {translated}")
        else:
            translated = text

        filename = "tts_output.mp3"
        tts = gTTS(text=translated, lang=lang_code)
        tts.save(filename)
        playsound(filename)
        if os.path.exists(filename):
            os.remove(filename)

    except Exception as e:
        print(f"[TTS] Error: {e}")


@app.route('/play/<lang>', methods=["GET", "POST"])
def play_voice(lang):
    """
    Called when user clicks English / Tamil / Hindi / Malayalam / Kannada button.
    Reads the current whole_word_text and speaks it in the chosen language.
    """
    if lang not in LANG_CODES:
        return f"Language '{lang}' not supported!", 400

    with state_lock:
        text_to_play = whole_word_text

    if not text_to_play.strip():
        print("[TTS] Buffer is empty — nothing to speak.")
        return render_template("home.html")

    # Run TTS in background thread so browser doesn't hang
    t = threading.Thread(
        target=generate_speech,
        args=(text_to_play, LANG_CODES[lang]),
        daemon=True
    )
    t.start()

    return render_template("home.html")


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 50)
    print("  Smart Sign Learning System")
    print("  Open browser → http://127.0.0.1:685")
    print("=" * 50)
    app.run(debug=True, port=685, use_reloader=False, threaded=True)
