# Prof.-X-s-Prize-House-Price-Prediction----1st-Winning-Team-Solution----92.6% Accuracy

# 🏠 House Price Prediction for Denver Dataset

This project builds and evaluates neural network models to predict house prices in Denver using the provided train and test datasets.
The work is implemented in PyTorch, with systematic preprocessing, training, validation, and comparison of model architectures.

# 📁 Dataset

Files: train.csv, test.csv

Goal: Predict the variable SALE_PRICE for the test data.

Key features include:
NBHD, PROP_CLASS, LIVING_SOFT, BSMT_SOFT, BSMT_AREA, LAND_SOFT, GRD_AREA, BLDG_AGE, FULL_B, HLP_B, STYLE_CN, etc.

The dataset is first explored through:

Histogram of Sale Prices — to observe price distribution.

Correlation Heatmap — showing strong relationships between SALE_PRICE, LIVING_SOFT, and FULL_B.

Neighborhood Distribution — analyzing property counts across neighborhoods.

# 🧹 Data Preprocessing

Steps performed before model training:

Feature Engineering

Derived area-based and ratio features (e.g., AREA_PER_BATH, LIVING_TO_LAND_RATIO).

Log-transformed variables to reduce skewness.

Handling Missing Values

Filled missing categorical data with "UNKNOWN".

Imputed numeric missing values with 0 after normalization.

Standardization

Numeric features normalized to mean = 0, std = 1.

Encoding Categorical Variables

Applied one-hot encoding for NBHD, PROP_CLASS, and STYLE_CN.

# 🔀 Data Splitting and Conversion

Split the preprocessed data into training and validation sets using train_test_split.

Converted pandas DataFrames to PyTorch tensors.

Created DataLoaders for efficient mini-batch training and validation.

🧩 Model Architectures
1. Linear Regression (Baseline)

A single-layer model used as a benchmark.

model = nn.Linear(in_features, 1)


Loss function: Mean Squared Error (MSE)
Evaluation metric: Median Error Rate (MER)

2. Multi-Layer Perceptron (Base Model)

Two hidden layers with ReLU activation:

model = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)


Optimized with Adam optimizer.

Trained for 200 epochs, tracking both training and validation MER.

3. Extended MLP (Deeper Network)

Model with four hidden layers (sizes 512, 256, 128, 64):

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


This deeper model achieved improved stability and lower validation error.

4. Regularization and Dropout

Added norm regularization and dropout layers to reduce overfitting.

Dropout applied after each fully connected layer with a probability of 0.2–0.3.

Improved generalization performance as shown by smaller training-validation error gaps.

# 📊 Training and Evaluation

Models trained for 200–400 epochs, depending on complexity.

Median Error Rate (MER) used for performance tracking.

Visualization:

Training vs Validation MER curves over epochs.

Clear convergence trends shown in plots.

# 💾 Model Saving and Inference

After training:

torch.save(model.state_dict(), 'checkpoint.pth')


Predictions generated for test data using:

model.eval()
with torch.no_grad():
    predicted_prices = model(test_features).numpy().flatten()


Predicted values were exponentiated (to reverse log transformation) and saved to submission.csv for Kaggle evaluation.

# 🧮 Key Takeaways

1. Model generalization is crucial. At the beginning of training, we were overly focused on achieving a new record low in validation error and neglected to compare the trends between training and validation losses. The best generalization typically occurs when the training loss is roughly equal to the validation loss, indicating a balanced model that neither underfits nor overfits.

2. Feature engineering matters a lot. It’s highly recommended to plot histograms of numerical and categorical features before feature engineering, so that transformations can be made with a clearer understanding of the data distribution. Moreover, not only independent variables but also the dependent variable can benefit from transformations — for instance, applying a log transform to a skewed target (as in this case) can stabilize training, though remember to inverse-transform predictions before reporting results.

3. Parameter tuning can be automated. Using tools like grid search or automated hyperparameter optimization saves a lot of manual effort. Without them, tuning feels like alchemy — time-consuming and exhausting.

4. Ensemble learning boosts performance. To further improve generalization, another effective strategy is inspired by the idea of Random Forest Model: ensemble learning through model averaging. By taking the best five individual models and averaging their predictions, we can achieve a more stable and accurate result. This ensemble approach significantly increases the chance of strong leaderboard performance — in fact, it was the key to our final winning solution on Kaggle.

5. Hardware matters too. If the budget allows, invest in a decent GPU. Training deep models on CPU-only machines is painfully slow — I learned this the hard way since my setup doesn’t support CUDA.

Thank you

Collaborators: Yanze (Ethan) Liu, Jiongyang (July) Song
