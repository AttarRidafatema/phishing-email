import re
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# PHISHING EMAIL DETECTION MODEL
# Mini Project using Machine Learning

DATASET = "emails.csv"
MODEL_FILE = "phishing_model.pkl"

URL_FEATURES = [
    "url_count",
    "max_url_length",
    "has_ip_address",
    "has_at_symbol",
    "has_http",
    "has_shortener",
    "suspicious_word_count"
]

SUSPICIOUS_WORDS = [
    "urgent", "verify", "password", "account",
    "suspended", "click", "login", "bank",
    "update", "confirm", "security", "free",
    "winner", "limited time", "immediately"
]

SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co",
    "goo.gl", "ow.ly", "is.gd"
]


# Convert dataset labels into 0 and 1
# 0 = Safe, 1 = Phishing

def clean_label(label):
    label = str(label).strip().lower()

    if label in ["phishing", "phish", "1", "1.0"]:
        return 1
    elif label in ["safe", "legitimate", "legit", "0", "0.0"]:
        return 0
    else:
        raise ValueError("Unknown label: " + label)


# Extract useful features from email text

def extract_features(email):
    email = str(email)
    text = email.lower()

    urls = re.findall(
        r"(?:https?://|www\.)[^\s<>\"']+",
        email,
        flags=re.IGNORECASE
    )

    urls = [
        url.rstrip(".,!?;:)]}")
        for url in urls
    ]

    url_lengths = [len(url) for url in urls]

    has_ip = 0
    has_at = 0
    has_http = 0
    has_shortener = 0

    for url in urls:
        check_url = url if "://" in url else "http://" + url
        parsed = urlparse(check_url)
        domain = parsed.hostname or ""

        if re.fullmatch(
            r"(?:\d{1,3}\.){3}\d{1,3}", domain
        ):
            has_ip = 1

        if "@" in url:
            has_at = 1

        if url.lower().startswith("http://"):
            has_http = 1

        if any(shortener in domain.lower()
               for shortener in SHORTENERS):
            has_shortener = 1

    suspicious_count = sum(
        text.count(word) for word in SUSPICIOUS_WORDS
    )

    return {
        "url_count": len(urls),
        "max_url_length": max(url_lengths, default=0),
        "has_ip_address": has_ip,
        "has_at_symbol": has_at,
        "has_http": has_http,
        "has_shortener": has_shortener,
        "suspicious_word_count": suspicious_count
    }


# Load and prepare the dataset

def load_dataset():
    print("\n[+] Loading email dataset...")

    df = pd.read_csv(DATASET)
    df.columns = df.columns.str.strip().str.lower()

    if not {"email_text", "label"}.issubset(df.columns):
        raise ValueError(
            "Dataset must contain email_text and label columns."
        )

    df = df[["email_text", "label"]].dropna()
    df["email_text"] = df["email_text"].astype(str)
    df["label"] = df["label"].apply(clean_label)

    df = df.drop_duplicates(subset=["email_text"])

    print("[+] Dataset loaded successfully!")
    print("[+] Total emails:", len(df))
    print("[+] Safe emails:", (df["label"] == 0).sum())
    print("[+] Phishing emails:", (df["label"] == 1).sum())

    if df["label"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both Safe and Phishing emails."
        )

    if df["label"].value_counts().min() < 2:
        raise ValueError(
            "At least two examples of each class are required."
        )

    return df


# Train the machine learning model

def train_model(df):
    print("\n[+] Extracting email features...")

    features = pd.DataFrame(
        df["email_text"].apply(extract_features).tolist(),
        index=df.index
    )

    X = pd.concat(
        [df[["email_text"]], features],
        axis=1
    )
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Convert email text into numerical features
    # and scale the extracted URL features

    preprocess = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=5000
                ),
                "email_text"
            ),
            (
                "url",
                StandardScaler(with_mean=False),
                URL_FEATURES
            )
        ]
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    print("\n[+] Training model...")
    model.fit(X_train, y_train)

    # Evaluate the model on unseen test data

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL RESULTS ==========")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Safe", "Phishing"],
        zero_division=0
    ))

    # Display confusion matrix

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Safe", "Phishing"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Phishing Email Detection")
    plt.tight_layout()
    plt.show()

    # Save the trained model

    joblib.dump(model, MODEL_FILE)
    print("\n[+] Model saved as", MODEL_FILE)

    return model


# Predict a new email

def predict_email(model):
    print("\n========== EMAIL PREDICTION ==========")

    email = input(
        "Paste the email text to check: "
    ).strip()

    if not email:
        print("[-] Email text cannot be empty.")
        return

    features = extract_features(email)

    new_email = pd.DataFrame([{
        "email_text": email,
        **features
    }])

    prediction = model.predict(new_email)[0]

    print("\n========== PREDICTION RESULT ==========")

    if prediction == 1:
        print("[!] Result: PHISHING EMAIL")
        print("[!] Be careful with links and requests.")
    else:
        print("[+] Result: SAFE EMAIL")
        print("[+] No phishing pattern detected by the model.")

    print("\nNote: This is a machine learning prediction,")
    print("not a guarantee that an email is safe.")


# Main program

def main():
    print("=" * 45)
    print("    PHISHING EMAIL DETECTION MODEL")
    print("          MINI PROJECT")
    print("=" * 45)

    try:
        df = load_dataset()
        model = train_model(df)

        while True:
            predict_email(model)

            choice = input(
                "\nCheck another email? (yes/no): "
            ).strip().lower()

            if choice != "yes":
                break

        print("\n[+] Program completed successfully!")

    except FileNotFoundError:
        print("\n[-] Dataset file not found!")
        print("Place emails.csv in the project folder.")
    except ValueError as error:
        print("\n[-] Dataset error:", error)
    except Exception as error:
        print("\n[-] Error:", error)


if __name__ == "__main__":
    main()