# CHEATSHEET — CLUSTERING (Machine Learning non supervisionato)

Usa questo schema quando l’esame richiede:
- clustering
- KMeans / Agglomerative / DBSCAN
- silhouette score
- encoding e scaling
- confronto con adjusted_rand_score

---

## 1. Caricamento ed esplorazione

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
df.shape
df.describe()
df.boxplot()
plt.show()
````

---

## 2. Pulizia e selezione feature

```python
df = df.dropna()
df = df.drop(columns=["id", "non_relevant"])
```

---

## 3. Encoding e scaling

```python
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

bin_cols = ["binary_col"]
cat_cols = ["categorical_col"]
num_cols = ["num1", "num2"]

df[bin_cols] = OrdinalEncoder().fit_transform(df[bin_cols])
df_cat = pd.get_dummies(df[cat_cols])
df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])

X = pd.concat([df[num_cols], df_cat], axis=1)
```

---

## 4. KMeans + silhouette

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(X)
    scores.append(silhouette_score(X, labels))
```

Scegli il k con silhouette migliore.

---

## 5. Clustering finale e distribuzione

```python
labels_km = KMeans(n_clusters=best_k).fit_predict(X)
pd.Series(labels_km).value_counts().plot(kind="bar")
plt.show()
```

---

## 6. Secondo clustering

```python
from sklearn.cluster import AgglomerativeClustering

labels_ag = AgglomerativeClustering(n_clusters=best_k).fit_predict(X)
```

---

## 7. Confronto clustering

```python
from sklearn.metrics import adjusted_rand_score

adjusted_rand_score(labels_km, labels_ag)
```

Commenta similarità e differenze.


---