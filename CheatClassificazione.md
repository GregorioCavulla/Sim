# CHEATSHEET — CLASSIFICAZIONE (Machine Learning supervisionato)

Usa questo schema quando l’esame richiede:
- target (classe)
- due modelli
- Cross Validation
- ottimizzazione di recall_macro o f1_macro
- confusion matrix e classification report

---

## 1. Caricamento ed esplorazione dati
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
df.shape
df.head()
df.describe()
df["target"].value_counts().plot(kind="bar")
plt.show()
````

---

## 2. Pulizia dati

```python
df = df.dropna()
df.shape
```

Oppure (se richiesto):

* numerici → media
* categorici → "unknown"

---

## 3. Split train / test

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
```

---

## 4. Modello 1 — tuning con CV

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

gs1 = GridSearchCV(
    DecisionTreeClassifier(random_state=0),
    param_grid,
    cv=cv,
    scoring="recall_macro"
)

gs1.fit(Xtr, ytr)
best1 = gs1.best_estimator_
```

---

## 5. Valutazione modello

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

yhat1 = best1.predict(Xte)
print(classification_report(yte, yhat1))

ConfusionMatrixDisplay.from_estimator(
    best1, Xte, yte, normalize="true"
)
plt.show()
```

---

## 6. Modello 2

Stesso identico schema del Modello 1, con un classificatore diverso.

---

## 7. Commento finale

* confronto metriche
* errori per classe
* modello migliore