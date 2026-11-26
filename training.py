# ============================================================
#  training.py  —  Fixed Model Training Script
#  Trains on ALL classes in hand.csv and saves hand_sign.pkl
# ============================================================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("="*50)
print("  SIGN LANGUAGE MODEL TRAINER")
print("="*50)

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv('hand.csv')
except FileNotFoundError:
    print("\nERROR: hand.csv not found!")
    print("Run preprocess.py first to collect training data.")
    exit()

print(f"\nLoaded hand.csv — {len(df)} total rows")
print("\nClass distribution:")
print(df['class'].value_counts().to_string())

unique_classes = df['class'].nunique()
print(f"\nTotal unique signs: {unique_classes}")

if unique_classes < 2:
    print("\n⚠  ERROR: You need AT LEAST 2 different sign classes to train a model!")
    print("   Run preprocess.py again and add more signs.")
    print("   Example signs to add: hello, good, bad, yes, no, thank you, help")
    exit()

# ── Prepare features ──────────────────────────────────────────────────────────
X = df.drop('class', axis=1)
y = df['class']

# Need enough samples per class for train/test split
min_samples = df['class'].value_counts().min()
if min_samples < 5:
    print(f"\n⚠  WARNING: Some classes have fewer than 5 samples (min={min_samples}).")
    print("   Consider recording more data for those signs.")
    test_size = 0.2
else:
    test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=1234, stratify=y
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── Train model ───────────────────────────────────────────────────────────────
# Using LogisticRegression — works better than RidgeClassifier for multi-class
print("\nTraining model...")
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Training complete!")
print(f"   Accuracy: {accuracy * 100:.1f}%")
print("\nDetailed report:")
print(classification_report(y_test, y_pred))

# ── Save model ────────────────────────────────────────────────────────────────
with open('hand_sign.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as hand_sign.pkl")
print("\nYou can now run:  python app.py")
