import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# ---------------------------
# 1. Load dataset
# ---------------------------
file_path = "data/Crime_Data_2010_2017.csv"
crime_df = pd.read_csv(file_path)


# ---------------------------
# 2. Clean & preprocess data
# ---------------------------
# Convert date and time columns
crime_df['Date Occurred'] = pd.to_datetime(crime_df['Date Occurred'], errors='coerce')
crime_df['Time Occurred'] = crime_df['Time Occurred'].astype(str).str.zfill(4)  # ensure 4-digit times
# Some rows may have non-numeric time after coercion to str; take first two chars and coerce to int safely
crime_df['Hour Occurred'] = pd.to_numeric(crime_df['Time Occurred'].str[:2], errors='coerce').fillna(0).astype(int)

# Drop rows with missing key info
crime_df = crime_df.dropna(subset=['Date Occurred', 'Area Name', 'Crime Code'])


# ---------------------------
# 3. Reduce number of crime categories
# ---------------------------
top_crimes = crime_df['Crime Code Description'].value_counts().head(30).index.tolist()
crime_df = crime_df[crime_df['Crime Code Description'].isin(top_crimes)]


# ---------------------------
# 4. Define spatio-temporal units
# ---------------------------
# ISO week number (isocalendar) returns a DataFrame-like object in pandas; use .week for compatibility
crime_df['Week'] = crime_df['Date Occurred'].dt.isocalendar().week
crime_df['Year'] = crime_df['Date Occurred'].dt.year

# Group by Area + Year + Week
grouped = crime_df.groupby(['Area Name', 'Year', 'Week'])


# ---------------------------
# 5. Aggregate labels per unit
# ---------------------------
units = []
for (area, year, week), group in grouped:
    label_vector = {crime: 0 for crime in top_crimes}
    for crime in group['Crime Code Description'].unique():
        label_vector[crime] = 1
    units.append({
        'Area Name': area,
        'Year': year,
        'Week': week,
        **label_vector
    })

unit_df = pd.DataFrame(units)
print(unit_df.head())

# Add simple contextual feature: previous week crimes
unit_df['Previous_Week_Crimes'] = unit_df.groupby('Area Name')[top_crimes].shift(1).sum(axis=1)
unit_df = unit_df.fillna(0)


# ---------------------------
# 6. Split train/test (temporal holdout)
# ---------------------------
train_df = unit_df[unit_df['Year'] <= 2015].reset_index(drop=True)
test_df = unit_df[unit_df['Year'] > 2015].reset_index(drop=True)

X_train = train_df.drop(columns=top_crimes).reset_index(drop=True)
y_train = train_df[top_crimes].reset_index(drop=True)

X_test = test_df.drop(columns=top_crimes).reset_index(drop=True)
y_test = test_df[top_crimes].reset_index(drop=True)


# ---------------------------
# 7. Encode categorical features
# ---------------------------
area_ohe = pd.get_dummies(X_train['Area Name'], prefix='Area')
X_train_enc = pd.concat([X_train.drop(columns=['Area Name', 'Year', 'Week']).reset_index(drop=True),
                         area_ohe.reset_index(drop=True)], axis=1)

area_ohe_test = pd.get_dummies(X_test['Area Name'], prefix='Area')
area_ohe_test = area_ohe_test.reindex(columns=area_ohe.columns, fill_value=0)
X_test_enc = pd.concat([X_test.drop(columns=['Area Name', 'Year', 'Week']).reset_index(drop=True),
                        area_ohe_test.reset_index(drop=True)], axis=1)

# Ensure consistent column ordering between train and test
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# Scale numeric features
numeric_cols = ['Previous_Week_Crimes']

# Impute missing numeric values (if any) before scaling to avoid sklearn errors
imputer = SimpleImputer(strategy='median')
X_train_enc[numeric_cols] = imputer.fit_transform(X_train_enc[numeric_cols])
X_test_enc[numeric_cols] = imputer.transform(X_test_enc[numeric_cols])

# Scale numeric features
scaler = StandardScaler()
X_train_enc[numeric_cols] = scaler.fit_transform(X_train_enc[numeric_cols])
X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])


# ---------------------------
# 8. Multi-label Logistic Regression (from-scratch implementation)
# ---------------------------
# We'll implement a simple binary logistic regression (gradient descent) and wrap one model per label.

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def binary_cross_entropy(y, y_pred):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


class BinaryLogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=800, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.verbose = verbose

    def fit(self, X, y):
        """
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        # initialize weights
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for i in range(self.n_iter):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear)

            # gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.verbose and (i % max(1, self.n_iter // 8) == 0):
                loss = binary_cross_entropy(y, y_pred)
                print(f"  Iter {i}/{self.n_iter} - loss: {loss:.4f}")

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return sigmoid(linear)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class MultiLabelLogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=800, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.models = {}      # dict: label -> BinaryLogisticRegressionScratch
        self.train_order = [] # keeps the order of labels trained
        self.verbose = verbose

    def fit(self, X, Y):
        """
        X: numpy array (n_samples, n_features)
        Y: pandas DataFrame of shape (n_samples, n_labels)
        """
        for label in Y.columns:
            # skip labels that are constant (all 0 or all 1) outside this function; caller can decide
            if self.verbose:
                print(f"Training label: {label}")
            model = BinaryLogisticRegressionScratch(lr=self.lr, n_iter=self.n_iter, verbose=self.verbose)
            model.fit(X, Y[label].values.astype(float))
            self.models[label] = model
            self.train_order.append(label)

    def predict_proba(self, X):
        """
        Returns numpy array shape (n_samples, n_labels) in the order self.train_order
        """
        probs = []
        for label in self.train_order:
            probs.append(self.models[label].predict_proba(X))
        if len(probs) == 0:
            return np.zeros((X.shape[0], 0))
        return np.column_stack(probs)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        if probs.shape[1] == 0:
            return np.zeros((X.shape[0], 0), dtype=int)
        return (probs >= threshold).astype(int)


# Some labels may be constant in the training set (all 0s or all 1s).
# LogisticRegression cannot be trained on a single-class target — detect and
# skip those labels, remembering their constant value so we can fill predictions.
label_cardinality = y_train.nunique()
train_labels = label_cardinality[label_cardinality > 1].index.tolist()
const_labels = label_cardinality[label_cardinality <= 1].index.tolist()

if len(train_labels) == 0:
    print("No trainable labels found (every label is constant in the training set). Skipping model fit.")
    trained = False
else:
    # Convert training features to numpy array
    X_train_np = X_train_enc.values.astype(float)
    X_test_np = X_test_enc.values.astype(float)

    # Instantiate and fit the scratch multi-label model only on trainable labels
    multi_lr_scratch = MultiLabelLogisticRegressionScratch(lr=0.1, n_iter=800, verbose=True)
    multi_lr_scratch.fit(X_train_np, y_train[train_labels])
    trained = True


# ---------------------------
# 9. Predict & Evaluate
# ---------------------------
# Build full predictions DataFrame, filling constant labels with their training value
if 'trained' in globals() and trained:
    # get predicted probabilities and binary preds for train_labels
    y_pred_proba_partial = multi_lr_scratch.predict_proba(X_test_np)  # shape (n_samples, n_train_labels)
    y_pred_partial = (y_pred_proba_partial >= 0.5).astype(int)

    # construct DataFrame for partial predictions (trainable labels)
    y_pred_df_partial = pd.DataFrame(y_pred_partial, columns=multi_lr_scratch.train_order, index=X_test_enc.index)
else:
    y_pred_df_partial = pd.DataFrame(index=X_test_enc.index)

# Construct the full prediction DataFrame
y_pred_df = pd.DataFrame(index=X_test_enc.index)
for col in train_labels:
    # ensure column order matches original train_labels list
    if col in y_pred_df_partial.columns:
        y_pred_df[col] = y_pred_df_partial[col]
    else:
        # Should not happen, but safe fallback
        y_pred_df[col] = 0

for col in const_labels:
    # fill with the constant value observed in training
    const_val = int(y_train[col].iloc[0])
    y_pred_df[col] = const_val

# Ensure column order matches top_crimes
y_pred_df = y_pred_df[top_crimes]

# Evaluate per-label. Use zero_division=0 to avoid warnings for labels with no positive predictions.
for crime in top_crimes:
    print(f"--- {crime} ---")
    y_true = y_test[crime].reset_index(drop=True)
    y_pred_col = y_pred_df[crime].reset_index(drop=True)
    # If the true labels in the test set are constant, sklearn's classification_report will still work
    print(classification_report(y_true, y_pred_col, zero_division=0))