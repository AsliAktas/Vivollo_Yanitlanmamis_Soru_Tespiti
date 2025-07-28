"""
06_diag_manual_tune.py
----------------------
• Manual hyperparameter tuning for LogisticRegression C values
• Reports train/val/test Macro-F1 for Cs=[0.2,0.5,1.0]

Usage:
    python src/06_diag_manual_tune.py
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

def load_data():
    base = Path(__file__).resolve().parent.parent
    train = pd.read_csv(base/'data'/'train.csv')
    val   = pd.read_csv(base/'data'/'val.csv')
    test  = pd.read_csv(base/'data'/'test.csv')
    return train, val, test

def encode_texts(texts, encoder):
    return encoder.encode(texts.tolist(), batch_size=64, show_progress_bar=False)

if __name__=='__main__':
    train, val, test = load_data()
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # prepare data
    X_train = encode_texts(train['context'].astype(str), encoder)
    y_train = train['kategori']
    X_val   = encode_texts(val['context'].astype(str), encoder)
    y_val   = val['kategori']
    X_test  = encode_texts(test['context'].astype(str), encoder)
    y_test  = test['kategori']

    for C in [0.2, 0.5, 1.0]:
        clf = LogisticRegression(C=C, max_iter=4000, class_weight='balanced', multi_class='multinomial', n_jobs=-1)
        clf.fit(X_train, y_train)
        print(f"C={C}")
        for name, X, y in [('train',X_train,y_train), ('val',X_val,y_val), ('test',X_test,y_test)]:
            pred = clf.predict(X)
            score = f1_score(y, pred, average='macro')
            print(f"  {name}: {score:.3f}")

