import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
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
crime_df['Hour Occurred'] = crime_df['Time Occurred'].str[:2].astype(int)

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

# Scale numeric features
scaler = StandardScaler()
numeric_cols = ['Previous_Week_Crimes']

# Impute missing numeric values (if any) before scaling to avoid sklearn errors
imputer = SimpleImputer(strategy='median')
X_train_enc[numeric_cols] = imputer.fit_transform(X_train_enc[numeric_cols])
X_test_enc[numeric_cols] = imputer.transform(X_test_enc[numeric_cols])

# Scale numeric features
X_train_enc[numeric_cols] = scaler.fit_transform(X_train_enc[numeric_cols])
X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])

# ---------------------------
# 8. Multi-label Logistic Regression
# ---------------------------
base_lr = LogisticRegression(max_iter=500)
multi_lr = MultiOutputClassifier(base_lr)

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
    # Fit only on labels that have at least two classes
    multi_lr.fit(X_train_enc, y_train[train_labels])
    trained = True

# ---------------------------
# 9. Predict & Evaluate
# ---------------------------
y_pred = multi_lr.predict(X_test_enc)
# Build full predictions DataFrame, filling constant labels with their training value
if 'trained' in globals() and trained:
    y_pred_partial = multi_lr.predict(X_test_enc)
    y_pred_df_partial = pd.DataFrame(y_pred_partial, columns=train_labels, index=X_test_enc.index)
else:
    y_pred_df_partial = pd.DataFrame(index=X_test_enc.index)

# Construct the full prediction DataFrame
y_pred_df = pd.DataFrame(index=X_test_enc.index)
for col in train_labels:
    y_pred_df[col] = y_pred_df_partial[col]
for col in const_labels:
    # fill with the constant value observed in training
    const_val = int(y_train[col].iloc[0])
    y_pred_df[col] = const_val

# Evaluate per-label. Use zero_division=0 to avoid warnings for labels with no positive predictions.
for crime in top_crimes:
    print(f"--- {crime} ---")
    y_true = y_test[crime].reset_index(drop=True)
    y_pred_col = y_pred_df[crime].reset_index(drop=True)
    # If the true labels in the test set are constant, sklearn's classification_report will still work
    print(classification_report(y_true, y_pred_col, zero_division=0))
