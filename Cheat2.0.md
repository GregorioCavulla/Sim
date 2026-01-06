Ottima idea. Aggiungendo questi pezzi completiamo il puzzle e copriamo quelle richieste specifiche che valgono i punti "extra" o "tecnici" negli esami che mi hai mostrato.

Ecco la versione **2.0 del Survival Kit**, integrata con il **Tuning (GridSearchCV)**, il **Preprocessing avanzato** (quello pignolo richiesto nei PDF) e la **Pulizia dati**.

Ho aggiunto una breve spiegazione prima di ogni blocco come richiesto, così sai *perché* lo stai usando.

---

# 🐍 Python Data Science Exam Cheat Sheet (Versione Completa)

## 1. Import e Pulizia Dati (La fase "Igienica") 🧹

*Prima ancora di guardare i dati, devi pulirli. Spesso nei file excel ci sono spazi nascosti nelle stringhe o valori mancanti che fanno crashare i modelli.*

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Caricamento
df = pd.read_csv('file.csv', sep=',') # Occhio al separatore (; o ,)

# 1. Pulizia Stringhe (Rimuove spazi vuoti a inizio/fine parola)
# Fondamentale se 'Low' viene letto come 'Low ' (con lo spazio)
df['colonna_testo'] = df['colonna_testo'].str.strip()

# 2. Gestione Valori Mancanti (NaN)
df.dropna(inplace=True)                 # Butta via le righe con buchi (più drastico)
# OPPURE
df.fillna(df.mean(), inplace=True)      # Riempi i buchi con la media (per numeri)
df.fillna('Unknown', inplace=True)      # Riempi i buchi con testo (per categorie)

```

## 2. Preprocessing Avanzato (Encoder & Scaler) ⚙️

*I modelli di Scikit-Learn vogliono SOLO numeri. Qui trasformi le parole in numeri nel modo "corretto" richiesto all'esame.*

```python
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder

# A. OrdinalEncoder: Per categorie con un ordine (es. Low < Medium < High)
# Crea un oggetto che sa che Low=0, Med=1, High=2
ord_enc = OrdinalEncoder(categories=[['Low', 'Med', 'High']])
df['grade_encoded'] = ord_enc.fit_transform(df[['grade']])

# B. OneHotEncoder: Per categorie SENZA ordine (es. Colore: Rosso, Blu)
# Trasforma la colonna in tante colonne di 0 e 1 (es. is_Red, is_Blue)
# drop='first' serve a evitare ridondanza (se non è rosso e non è blu, è verde)
df = pd.get_dummies(df, columns=['colore'], drop_first=True)

# C. MinMaxScaler: Schiaccia tutti i numeri tra 0 e 1
# Utile per il Clustering o se hai variabili con scale diverse (es. Età vs Stipendio)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

```

## 3. Tuning dei Parametri (GridSearchCV) 🎛️

*Questa è la funzione "magica" che prova tante combinazioni per te e trova il modello migliore. Ti salva dal dover provare a mano.*

```python
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# 1. Definisci la griglia: quali parametri vuoi testare?
param_grid = {
    'max_depth': [3, 5, 10, None],          # Profondità albero
    'criterion': ['gini', 'entropy'],       # Criterio di divisione
    'min_samples_split': [2, 5]             # Minimo campioni per dividere un nodo
}

# 2. Configura la ricerca
# cv=5 significa Cross Validation a 5 fold (robustezza)
# scoring='recall_macro' o 'accuracy' o 'f1_macro' (leggi cosa chiede il prof!)
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='recall_macro')

# 3. Addestra (Lui farà decine di prove qui)
grid.fit(X_train, y_train)

# 4. Risultati
print("Migliori parametri:", grid.best_params_)
best_model = grid.best_estimator_ # Questo è il modello già pronto e addestrato

```

## 4. Valutazione Dettagliata (Il "Voto" al Modello) 📝

*Oltre alla semplice accuratezza, questi comandi ti danno un quadro completo per prendere il massimo dei voti.*

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Usa il modello migliore trovato dalla GridSearch
y_pred = best_model.predict(X_test)

# Report completo (Precision, Recall, F1-score per ogni classe)
print(classification_report(y_test, y_pred))

# Matrice di Confusione Normalizzata (chiesta spesso negli esami)
# normalize='true' mostra le percentuali invece dei numeri assoluti
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, normalize='true')
plt.show()

```

## 5. Clustering & Association Rules (I "Soliti Sospetti") 🛒

*Questi rimangono uguali alla versione precedente, ma ricorda l'importante distinzione:*

* **Clustering (KMeans):** Usa `silhouette_score` per valutare.
* **Association Rules:** Usa `mlxtend` (NON Scikit-learn). Ricorda `apriori` prima, `association_rules` dopo.

