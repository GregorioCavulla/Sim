
# CHEATSHEET — ASSOCIATION RULES / APRIORI

Usa questo schema quando l’esame richiede:
- database transazionale
- InvoiceNo / Description
- basket matrix
- apriori
- lift, confidence, support

---

## 1. Caricamento dati
```python
import pandas as pd

df = pd.read_excel("Online-Retail-France.xlsx")
df.shape
df.head()
````

---

## 2. Pulizia dati

```python
df["Description"] = df["Description"].str.strip()
df = df.dropna(subset=["InvoiceNo"])
df = df[~df["InvoiceNo"].str.contains("C")]
df = df[~df["Description"].str.contains("POSTAGE")]
```

---

## 3. Basket matrix

```python
basket = (
    df.groupby(["InvoiceNo", "Description"])["Quantity"]
      .sum().unstack().reset_index()
      .fillna(0).set_index("InvoiceNo")
)
```

---

## 4. Binarizzazione

```python
def encode(x):
    return x > 0

basket_bool = basket.map(encode)
```

---

## 5. Ricerca del min_support

```python
from mlxtend.frequent_patterns import apriori, association_rules

min_support = 1.0
while min_support > 0:
    freq = apriori(
        basket_bool,
        min_support=min_support,
        use_colnames=True
    )
    rules = association_rules(
        freq,
        metric="lift",
        min_threshold=1
    )
    if rules.shape[0] >= 20:
        break
    min_support -= 0.01
```

---

## 6. Ordinamento regole

```python
rules = rules.sort_values(
    by=["lift", "confidence"],
    ascending=False
)
```

---

## 7. Scatter plot

```python
import matplotlib.pyplot as plt

plt.scatter(rules["confidence"], rules["lift"])
plt.xlabel("confidence")
plt.ylabel("lift")
plt.show()
```