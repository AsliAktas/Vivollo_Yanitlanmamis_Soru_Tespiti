# src/plot_confusion_matrix.py
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
root = Path(__file__).resolve().parent.parent
val = pd.read_csv(root/'data'/'val.csv')
bundle = joblib.load(root/'models'/'cat_model'/'model.joblib')
encoder, clf = bundle['encoder'], bundle['clf']

# Prepare data
X_val = val['context'].tolist()
y_true = val['kategori'].tolist()
emb = encoder.encode(X_val, batch_size=32)
y_pred = clf.predict(emb)

# Plot
cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Category Confusion Matrix (Validation)')
plt.savefig('outputs/confusion_matrix_validation.png', dpi=200)
plt.show()
