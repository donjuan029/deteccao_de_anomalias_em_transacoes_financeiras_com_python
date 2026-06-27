# 🔍 Detecção de Anomalias em Transações Financeiras com Python

> Pipeline completo de Machine Learning para detecção de fraude em cartões de crédito — do EDA à explicabilidade com SHAP.

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Dataset](#dataset)
- [Estrutura do Notebook](#estrutura-do-notebook)
- [Técnicas Utilizadas](#técnicas-utilizadas)
- [Modelos Treinados](#modelos-treinados)
- [Resultados](#resultados)
- [Como Executar](#como-executar)
- [Dependências](#dependências)
- [Lições Técnicas](#lições-técnicas)

---

## Sobre o Projeto

Projeto de detecção de fraude em transações de cartão de crédito utilizando técnicas de Machine Learning supervisionado. O foco está na **qualidade do pipeline** — com atenção especial a data leakage, balanceamento de classes e explicabilidade do modelo.

Desenvolvido como projeto de estudo e portfólio na plataforma **DIO (Digital Innovation One)**.

---

## Dataset

| Atributo | Valor |
|---|---|
| Fonte | [TensorFlow / Kaggle — Credit Card Fraud Detection](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv) |
| Instâncias | 284.807 transações |
| Features | 30 (V1–V28 via PCA + Amount + Time) |
| Classes | 0 = Normal · 1 = Fraude |
| Desbalanceamento | ~99.83% Normal · ~0.17% Fraude |

> As features V1–V28 são componentes principais (PCA) aplicados para anonimizar os dados originais. Apenas `Amount` e `Time` permanecem em escala original.

---

## Estrutura do Notebook

```
01 — Importação e Carregamento
02 — Análise Exploratória (EDA)
      ├── Distribuição das classes
      ├── Distribuição do Amount por classe
      ├── Distribuição do Time por classe
      ├── Boxplots Amount e Time
      ├── Heatmap de correlação
      └── Histogramas V1–V10 por classe
03 — Tratamento de Dados
      ├── df.info()
      ├── df.describe()
      ├── Valores nulos
      └── Duplicatas
04 — Feature Engineering
      ├── Amount_log (log1p para reduzir skewness)
      └── Hour (ciclo temporal derivado de Time)
05 — Train/Test Split + SMOTE (correto)
06 — Treinamento dos Modelos
      ├── Logistic Regression (Pipeline)
      ├── Random Forest (Pipeline)
      └── XGBoost (Pipeline)
07 — Validação Cruzada (StratifiedKFold 5-fold)
08 — GridSearch Aprimorado (RandomizedSearchCV)
09 — Comparativo Final dos Modelos
10 — Explicabilidade com SHAP
      ├── Bar Plot (importância global)
      ├── Beeswarm (distribuição de impacto)
      └── Waterfall (explicação individual)
```

---

## Técnicas Utilizadas

### Balanceamento de Classes
- **SMOTE (Synthetic Minority Over-sampling Technique)**: gera amostras sintéticas da classe minoritária
- Aplicado **somente no conjunto de treino**, dentro de cada fold — evitando data leakage

### Prevenção de Data Leakage
Dois pontos críticos tratados neste projeto:

```
❌ Errado                          ✅ Correto
─────────────────────────────────────────────────────────
scaler.fit_transform(df[...])     Pipeline com scaler interno
SMOTE(x, y) antes do split        SMOTE só no X_train após split
```

Ambos resolvidos via `imblearn.pipeline.Pipeline`, garantindo que o `StandardScaler` e o `SMOTE` sejam fitados apenas nos dados de treino de cada fold.

### Validação
- `StratifiedKFold (k=5)` preserva a proporção de classes em cada fold
- `RandomizedSearchCV` para tuning eficiente com grids grandes

### Métricas de Avaliação

| Métrica | Por que usar em fraude |
|---|---|
| **Recall (Fraude)** | Minimiza falsos negativos (fraudes não detectadas) |
| **Precision (Fraude)** | Controla falsos positivos (bloqueios indevidos) |
| **F1-Score** | Equilíbrio entre Precision e Recall |
| **AUC-ROC** | Performance geral do ranking de probabilidades |
| **Average Precision** | AUC da curva PR — mais informativa que ROC em dados desbalanceados |

---

## Modelos Treinados

| Modelo | Configuração Principal |
|---|---|
| **Logistic Regression** | `class_weight="balanced"`, `max_iter=1000` |
| **Random Forest** | `n_estimators=100`, `max_depth=10`, `class_weight="balanced"` |
| **XGBoost** | `scale_pos_weight=10`, `eval_metric="logloss"` |
| **XGBoost + GridSearch** | `RandomizedSearchCV` com 7 hiperparâmetros, 20 iterações |

---

## Resultados

> Os valores abaixo são referência do dataset completo. Execute o notebook para obter os resultados exatos do seu ambiente.

| Modelo | AUC-ROC | Recall (Fraude) | F1 (Fraude) |
|---|---|---|---|
| Logistic Regression | ~0.97 | ~0.72 | ~0.74 |
| Random Forest | ~0.97 | ~0.80 | ~0.85 |
| XGBoost | ~0.98 | ~0.82 | ~0.87 |
| XGBoost (GridSearch) | ~0.98 | ~0.84 | ~0.88 |

**Conclusão**: XGBoost com tuning apresenta o melhor equilíbrio entre Recall e Precision para a classe de fraude. O ajuste de threshold (padrão 0.5 → 0.3) aumenta o Recall em troca de mais falsos positivos — decisão que depende do custo de negócio.

---

## Como Executar

### Google Colab (recomendado)
```
1. Acesse: https://colab.research.google.com
2. File → Upload notebook → selecione o .ipynb
3. Runtime → Run all
```

### Local
```bash
# Clone o repositório
git clone https://github.com/donjuan029/<repo-name>.git
cd <repo-name>

# Instale as dependências
pip install -r requirements.txt

# Execute o Jupyter
jupyter notebook Deteccao_Anomalias_Transacoes_v2.ipynb
```

---

## Dependências

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
shap>=0.42.0
```

> Instalar tudo: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap`

---

## Lições Técnicas

### ⚠️ Data Leakage — o erro mais comum

**SMOTE antes do split:**
```python
# ❌ Errado — cria amostras sintéticas com dados do test set
smote.fit_resample(X, y)  # X inclui test set

# ✅ Correto — SMOTE só no treino
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

**Scaler fora do Pipeline:**
```python
# ❌ Errado — scaler vê o test set durante o fit
scaler.fit_transform(df[["Amount"]])  # antes do split

# ✅ Correto — scaler fitado apenas no treino de cada fold
Pipeline([("scaler", StandardScaler()), ("clf", modelo)])
```

### 📌 Accuracy não é métrica para fraude

Com 99.83% de transações normais, um modelo que classifica **tudo como normal** tem 99.83% de accuracy — e detecta 0% das fraudes. Por isso o foco está em **Recall**, **F1** e **AUC-PR**.

---

## Autor

**Desenvolvido pela DIO (Digital Innovation One) e atualizado por Juan Carlo Andrade Cruz**

---

*Projeto desenvolvido para o bootcamp DIO — Detecção de Anomalias em Transações com Python*
