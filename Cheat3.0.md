Certamente. Ho smontato i "blocchi giganti" e ho creato un flusso logico funzione per funzione.

Questo formato è pensato per essere letto riga per riga: **Funzione -> A cosa serve -> Codice**.
I blocchi "indivisibili" (come la creazione del Basket o il GridSearch) sono stati mantenuti uniti ma commentati riga per riga.

Ecco il tuo **Dizionario delle Funzioni da Esame**.

---

# 🐍 PYTHON DATA SCIENCE: FUNCTION-BY-FUNCTION CHEATSHEET

## 1. IMPORT & CARICAMENTO 📥

**Librerie Base**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

```

**Caricare CSV**
*Nota: controlla sempre se il separatore è virgola `,` o punto e virgola `;` aprendo il file con notepad.*

```python
df = pd.read_csv('nome_file.csv', sep=',') 

```

**Caricare Excel**
*Nota: serve per gli esami tipo "Retail".*

```python
df = pd.read_excel('nome_file.xlsx')

```

## 2. ISPEZIONE DATI (I primi comandi da lanciare) 🧐

**Visione d'insieme**
*Mostra righe, colonne, tipi di dati e memoria usata.*

```python
df.info()

```

**Statistiche rapide**
*Media, deviazione standard, min, max per le colonne numeriche.*

```python
df.describe()

```

**Controllo dimensioni**
*(Righe, Colonne).*

```python
print(df.shape)

```

**Lista nomi colonne**
*Utile per fare copia-incolla dei nomi senza sbagliare a scriverli.*

```python
print(df.columns)

```

## 3. PULIZIA DATI (Obbligatorio per non avere errori) 🧹

**Pulizia spazi nelle stringhe**
*Fondamentale: rimuove spazi invisibili (es. "France " diventa "France").*

```python
df['Colonna'] = df['Colonna'].str.strip()

```

**Contare valori nulli**
*Ti dice quanti buchi ci sono per ogni colonna.*

```python
print(df.isnull().sum())

```

**Eliminare righe con nulli**
*`inplace=True` applica la modifica subito senza dover scrivere df = ...*

```python
df.dropna(inplace=True)

```

**Eliminare righe specifiche (Filtro NOT)**
*Es. Tieni tutto ciò che NON contiene 'C' (ordini cancellati).*

```python
df = df[~df['InvoiceNo'].astype(str).str.contains('C')]

```

**Riempire nulli (Imputazione)**
*Se richiesto di non cancellare ma riempire (es. con la media).*

```python
df['Age'].fillna(df['Age'].mean(), inplace=True)

```

## 4. VISUALIZZAZIONE (EDA) 📊

**Istogramma Distribuzione**
*Per vedere come è spalmata una variabile (es. Prezzi).*

```python
df['Prezzo'].hist(bins=30)
plt.show()

```

**Boxplot (Caccia agli Outlier)**
*Il baffo mostra il range, i puntini fuori sono outlier.*

```python
sns.boxplot(x='Classe', y='Valore', data=df)
plt.show()

```

**Pairplot (Visione Totale)**
*Incrocia tutte le variabili. `hue` colora in base alla classe.*

```python
sns.pairplot(df, hue='Target')
plt.show()

```

**Matrice di Correlazione**
*Per vedere quali variabili numeriche sono collegate.*

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

```

## 5. PREPROCESSING (Trasformazione Numerica) ⚙️

**Separazione X (Dati) e y (Target)**

```python
X = df.drop('Target', axis=1) # Tutte le colonne tranne il target
y = df['Target']              # Solo la colonna target

```

**Divisione Train / Test**
*`stratify=y` mantiene le proporzioni delle classi (es. 70% sì, 30% no).*

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

```

**Encoding Ordinale (C'è un ordine)**
*Trasforma Low->0, Med->1, High->2.*

```python
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[['Low', 'Med', 'High']])
df['Grado'] = encoder.fit_transform(df[['Grado']])

```

**Encoding One-Hot (Non c'è ordine)**
*Trasforma Colore in: Is_Red, Is_Blue (0/1). `drop_first` evita ridondanze.*

```python
df = pd.get_dummies(df, columns=['Colore'], drop_first=True)

```

**Scaling (Normalizzazione)**
*Porta tutti i numeri tra 0 e 1. Obbligatorio per Clustering e KNN.*

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train) # Fit solo sul train!
X_test = scaler.transform(X_test)       # Solo transform sul test

```

**Selezione Attributi (Feature Selection)**
*Tiene solo le K colonne migliori.*

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(score_func=mutual_info_classif, k=5)
X_new = selector.fit_transform(X, y)

```

## 6. CLASSIFICAZIONE & TUNING (Supervisionato) 🌲

**Definizione Strategia Cross-Validation**
*Divide i dati in 5 parti rispettando le proporzioni.*

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

```

**Griglia Iperparametri**
*Dizionario con le opzioni da provare.*

```python
param_grid = {
    'max_depth': [3, 5, 10], 
    'criterion': ['gini', 'entropy']
}

```

**GridSearchCV (Il Cuore del Tuning)**
*Prova tutte le combinazioni. `scoring` può essere 'accuracy', 'recall_macro', 'f1_macro'.*

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=cv, scoring='recall_macro')
grid.fit(X_train, y_train)

```

**Recupero Miglior Modello**

```python
best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)

```

**Valutazione (Report)**
*Stampa Precision, Recall, F1 per ogni classe.*

```python
from sklearn.metrics import classification_report
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

```

**Matrice di Confusione Normalizzata**
*Mostra le percentuali di errore.*

```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, normalize='true')
plt.show()

```

## 7. CLUSTERING (Non Supervisionato) 🎯

**Ricerca K Ottimale (Elbow Method)**
*Ciclo obbligatorio per giustificare la scelta di K.*

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sil = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled) # Usa dati scalati!
    sil.append(silhouette_score(X_scaled, km.labels_))

# Plot per scegliere K (picco più alto)
plt.plot(range(2, 11), sil, marker='o')
plt.show()

```

**Esecuzione Clustering Finale**
*Supponendo che dal grafico K=3 sia il migliore.*

```python
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

```

**Visualizzazione Cluster**
*Aggiungi le etichette al dataset originale e guarda.*

```python
df['Cluster'] = labels
sns.pairplot(df, hue='Cluster')
plt.show()

```

## 8. ASSOCIATION RULES (Market Basket) 🛒

*Attenzione: qui si usa la libreria `mlxtend`.*

**Creazione Basket Matrix (Blocco Unico)**
*Trasforma la lista transazioni in matrice Prodotti x Scontrini. Impara a memoria la sequenza: groupby -> sum -> unstack -> reset -> fillna -> set_index.*

```python
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum()
          .unstack()
          .reset_index()
          .fillna(0)
          .set_index('InvoiceNo'))

```

**Binarizzazione**
*Converte quantità (10, 20) in presenza (True/False).*

```python
basket_bool = basket.map(lambda x: x > 0)

```

**Ciclo While per Supporto (Blocco Unico)**
*Richiesta tipica: "Trova supporto per avere almeno 20 regole".*

```python
from mlxtend.frequent_patterns import apriori, association_rules

min_sup = 0.1 # Valore iniziale alto
while min_sup > 0.001:
    # 1. Trova itemset
    frequent = apriori(basket_bool, min_support=min_sup, use_colnames=True)
    # 2. Genera regole
    if not frequent.empty:
        rules = association_rules(frequent, metric="lift", min_threshold=1)
        # 3. Controlla numero regole
        if len(rules) >= 20:
            print(f"Trovato! Min Sup: {min_sup}")
            break
    min_sup -= 0.01 # Decrementa

```

**Visualizzazione Regole**

```python
print(rules.sort_values('lift', ascending=False).head(5))

```

## 9. REGRESSIONE (Numerica) 📈

**Linear Regression Semplice**

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

```

**Metriche Regressione**
*R2 vicino a 1 = Bene. RMSE basso = Bene.*

```python
from sklearn.metrics import r2_score, mean_squared_error
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

```