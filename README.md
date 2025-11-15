# 🧠 Prof.-X’s Prize: House Price Prediction (🥇 1st Winning Team)

## 🏠 Overview

This project builds and evaluates **neural network models** to predict **house prices in Denver** using the provided **train** and **test** datasets.  
It is fully implemented in **PyTorch**, featuring systematic **data preprocessing**, **model training**, **validation**, and **architecture comparison**.

---

## 📁 Dataset

**Files:**  
- `train.csv`  
- `test.csv`

**Goal:** Predict the target variable `SALE_PRICE` for the test dataset.

**Key features include:**  
`NBHD`, `PROP_CLASS`, `LIVING_SOFT`, `BSMT_SOFT`, `BSMT_AREA`, `LAND_SOFT`, `GRD_AREA`, `BLDG_AGE`, `FULL_B`, `HLP_B`, `STYLE_CN`, etc.

**Exploratory Analysis:**
- **Histogram of Sale Prices** — to visualize price distribution.  
- **Correlation Heatmap** — reveals strong relationships between `SALE_PRICE`, `LIVING_SOFT`, and `FULL_B`.  
- **Neighborhood Distribution** — shows how property counts vary across neighborhoods.

---

## 🧹 Data Preprocessing

Steps performed before model training:

### 1. **Feature Engineering**
- Derived new ratio and area-based features (e.g., `AREA_PER_BATH`, `LIVING_TO_LAND_RATIO`).
- Applied **log transformation** to reduce skewness in highly non-linear variables.

### 2. **Handling Missing Values**
- Filled missing categorical values with `"UNKNOWN"`.
- Replaced missing numeric values with `0` (after normalization).

### 3. **Standardization**
- Scaled numerical features to **mean = 0**, **standard deviation = 1**.

### 4. **Encoding Categorical Variables**
- Applied **one-hot encoding** for `NBHD`, `PROP_CLASS`, and `STYLE_CN`.

---

## 🔀 Data Splitting and Conversion

- Split the preprocessed dataset into **training** and **validation** sets using `train_test_split`.  
- Converted pandas DataFrames into **PyTorch tensors**.  
- Created **DataLoaders** to support efficient mini-batch training and validation.

---

## 🧩 Model Architectures

### 1. **Linear Regression (Baseline)**
A simple benchmark model:
```python
model = nn.Linear(in_features, 1)
```

- **Loss Function:** Mean Squared Error (MSE)  
- **Metric:** Median Error Rate (MER)

---

### 2. **Multi-Layer Perceptron (Base Model)**
Two hidden layers with ReLU activation:
```python
model = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
```

- **Optimizer:** Adam  
- **Training Duration:** 200 epochs  
- **Metrics Tracked:** Training and validation MER

---

### 3. **Extended MLP (Deeper Network)**
A deeper network achieving better stability and lower validation error:
```python
model = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
```

---

### 4. **Regularization and Dropout**
- Added **L2 norm regularization** and **dropout layers** (p = 0.2–0.3) to mitigate overfitting.  
- Improved generalization performance shown by a reduced gap between training and validation losses.

---

## 📊 Training and Evaluation

- Models trained for **200–400 epochs**, depending on complexity.  
- **Median Error Rate (MER)** used for evaluation.  
- Visualization: **training vs validation MER curves** demonstrate convergence and stability.

---

## 💾 Model Saving and Inference

After training:
```python
torch.save(model.state_dict(), 'checkpoint.pth')
```

Generating predictions for test data:
```python
model.eval()
with torch.no_grad():
    predicted_prices = model(test_features).numpy().flatten()
```

Reverse the log transformation and save results:
```python
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": np.expm1(predicted_prices)
})
submission.to_csv("submission.csv", index=False)
```

---

## 🧮 Key Takeaways

1. **Model Generalization Matters**  
   At the beginning of training, we were overly focused on achieving a new record low in validation error and neglected to compare the trends between training and validation losses.
   The best generalization typically occurs when the training loss is roughly equal to the validation loss, indicating a balanced model that neither underfits nor overfits.

2. **Feature Engineering Is Critical**  
   It’s highly recommended to plot histograms of numerical and categorical features before feature engineering, so that transformations can be made with a clearer understanding of the data distribution.
   Moreover, not only independent variables but also the dependent variable can benefit from transformations — for instance, applying a log transform to a skewed target (as in this case) can stabilize training,
   though remember to inverse-transform predictions before reporting results.

3. **Automate Hyperparameter Tuning**  
   Using tools like grid search or automated hyperparameter optimization saves a lot of manual effort. Without them, tuning feels like alchemy — time-consuming and exhausting.

5. **Ensemble Learning Boosts Accuracy**  
   To further improve generalization, another effective strategy is inspired by the idea of Random Forest Model: ensemble learning through model averaging.
   By taking the best five individual MLP models with different parameters and averaging their predictions, we can achieve a more stable and accurate result.
   This ensemble approach significantly increases the chance of strong leaderboard performance — in fact, it was the key to our final winning solution on Kaggle.

7. **Hardware Matters**  
   If the budget allows, invest in a decent GPU. Training deep models on CPU-only machines is painfully slow — I learned this the hard way since my setup doesn’t support CUDA.

   

---

## 👥 Collaborators

**Yanze (Ethan) Liu**  
**Jiongyang (July) Song**
