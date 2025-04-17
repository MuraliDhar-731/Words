# train_model.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/LinkedInLearning/dsm-bank-model-2870047/main/bankData/bank.csv"
    df = pd.read_csv(url)
    df.drop(df.iloc[:, 8:16], inplace=True, axis=1)

    # Separate numeric and categorical
    X_categoric = df.iloc[:, [1, 2, 3, 4, 6, 7]].values
    X_numeric = df.iloc[:, [0, 5]].values
    y = df.iloc[:, -1].values

    # One-hot encode categorical
    ohe = OneHotEncoder()
    categoric_data = ohe.fit_transform(X_categoric).toarray()
    categoric_columns = ohe.get_feature_names_out()
    categoric_df = pd.DataFrame(categoric_data, columns=categoric_columns)

    # Standardize numeric
    scaler = StandardScaler()
    numeric_df = pd.DataFrame(scaler.fit_transform(X_numeric), columns=["age", "balance"])

    # Combine features
    X_final = pd.concat([numeric_df, categoric_df], axis=1)
    X_final.columns = X_final.columns.astype(str)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X_final, y_encoded

def train_and_tune_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'class_weight': [None, 'balanced']
    }

    rfc = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(rfc, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("✅ Best Parameters:", grid_search.best_params_)
    print(f"📈 Accuracy: {accuracy:.4f}")
    print("📋 Classification Report:\n", report)

    # Save the trained model
    joblib.dump(best_model, "bank_model_optimized.pkl")
    print("💾 Model saved as 'bank_model_optimized.pkl'")

if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    train_and_tune_model(X, y)
