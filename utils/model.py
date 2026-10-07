import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def train_model(texts, labels, test_size=0.2, random_state=42,
                max_features=5000, ngram_range=(1, 2), min_df=1):
    texts = texts.fillna("").astype(str)
    labels = labels.fillna("").astype(str).str.strip()

    if len(texts) < 2:
        raise ValueError("Dataset terlalu sedikit untuk training.")
    if labels.nunique() < 2:
        raise ValueError("Naive Bayes membutuhkan minimal dua kelas.")
    if (texts.str.strip() == "").all():
        raise ValueError("Semua teks preprocessing kosong. Periksa dataset/preprocessing.")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(texts)
    if X.shape[1] == 0:
        raise ValueError("TF-IDF tidak menghasilkan fitur. Periksa teks dataset.")

    # First attempt: stratified split as requested.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    except ValueError as exc:
        # Fallback for very small/imbalanced datasets.
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=test_size, random_state=random_state, stratify=None
            )
        except ValueError as exc2:
            raise ValueError(
                "Train-test split gagal. Tambahkan data dan pastikan terdapat minimal dua kelas dengan jumlah yang memadai."
            ) from exc2

    if len(set(y_train)) < 2:
        raise ValueError(
            "Data training hanya memiliki satu kelas setelah split. Tambahkan data per kelas atau gunakan dataset lebih seimbang."
        )

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "vectorizer": vectorizer,
        "model": clf,
        "X_train": X_train,
        "X_test": X_test,
        "X_all": X,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "labels": list(clf.classes_),
    }


def evaluate_model(y_test, y_pred):
    labels = list(dict.fromkeys(list(y_test) + list(y_pred)))
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "report": report,
    }


def predict_text(text, vectorizer, model):
    from utils.preprocessing import preprocess_text

    cleaned = preprocess_text(text)
    if not cleaned:
        raise ValueError("Ulasan menjadi kosong setelah preprocessing.")

    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        probabilities = {
            cls: float(prob) for cls, prob in zip(model.classes_, probs)
        }

    return pred, probabilities


def get_top_tfidf_words(vectorizer, X_all, n=20):
    if X_all is None or X_all.shape[0] == 0:
        return []

    scores = np.asarray(X_all.mean(axis=0)).ravel()
    features = vectorizer.get_feature_names_out()
    if len(features) == 0:
        return []

    indices = np.argsort(scores)[::-1][:n]
    return [(features[i], float(scores[i])) for i in indices if scores[i] > 0]
