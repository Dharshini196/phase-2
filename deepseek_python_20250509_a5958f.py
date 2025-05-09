# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import joblib
import shap

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
def load_data(filepath):
    """Load customer churn dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    return df

# Exploratory Data Analysis
def perform_eda(df):
    """Perform exploratory data analysis"""
    print("\n=== Basic Dataset Info ===")
    print(df.info())
    
    print("\n=== Summary Statistics ===")
    print(df.describe(include='all'))
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Visualize target variable distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.show()
    
    # Visualize numerical features
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols].hist(figsize=(15, 10))
    plt.suptitle('Numerical Features Distribution')
    plt.show()
    
    # Visualize correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.show()

# Feature Engineering
def preprocess_data(df):
    """Preprocess and engineer features"""
    # Handle missing values (example - adjust based on your data)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Create new features (example features)
    df['TenureToAgeRatio'] = df['tenure'] / (df['SeniorCitizen'] + 1)
    df['MonthlyChargesToTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity', 
                              'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                              'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)
    
    # Convert target variable to binary
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return df

# Data Preparation
def prepare_data(df):
    """Prepare data for modeling"""
    # Define features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Define categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

# Model Training
def train_models(X_train, y_train, preprocessor):
    """Train multiple machine learning models"""
    # Define models and their parameter grids for tuning
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, class_weight='balanced'),
            'params': {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__solver': ['liblinear', 'saga']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, class_weight='balanced', probability=True),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }
        }
    }
    
    best_models = {}
    
    for model_name, model_info in models.items():
        print(f"\n=== Training {model_name} ===")
        
        # Create pipeline with SMOTE for handling class imbalance
        pipeline = make_imb_pipeline(
            preprocessor,
            SMOTE(random_state=42),
            model_info['model']
        )
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid=model_info['params'],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store the best model
        best_models[model_name] = {
            'model': grid_search.best_estimator_,
            'params': grid_search.best_params_,
            'score': grid_search.best_score_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV AUC score: {grid_search.best_score_:.4f}")
    
    return best_models

# Model Evaluation
def evaluate_models(best_models, X_test, y_test):
    """Evaluate models on test set"""
    results = []
    
    for model_name, model_info in best_models.items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })
        
        # Print classification report
        print(f"\n=== {model_name} Classification Report ===")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), 
                    annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Churn', 'Churn'],
                    yticklabels=['Not Churn', 'Churn'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.show()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
        
        # Plot Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label=model_name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
    
    # Display results dataframe
    results_df = pd.DataFrame(results)
    print("\n=== Model Performance Comparison ===")
    print(results_df.sort_values(by='ROC AUC', ascending=False))
    
    return results_df

# Feature Importance Analysis
def analyze_feature_importance(best_model, X_train, preprocessor):
    """Analyze feature importance using SHAP values"""
    # Process the data to get feature names
    preprocessor.fit(X_train)
    
    # Get feature names after one-hot encoding
    categorical_features = X_train.select_dtypes(include=['object']).columns
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Get one-hot encoded feature names
    ohe = preprocessor.named_transformers_['cat']
    ohe_feature_names = ohe.get_feature_names_out(categorical_features)
    
    # Combine all feature names
    all_feature_names = numerical_features.tolist() + ohe_feature_names.tolist()
    
    # Create SHAP explainer
    explainer = shap.Explainer(best_model.named_steps['classifier'])
    
    # Transform the training data
    X_train_processed = preprocessor.transform(X_train)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_train_processed)
    
    # Plot summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train_processed, feature_names=all_feature_names, plot_type='bar')
    plt.title('Feature Importance (SHAP Values)')
    plt.show()
    
    # Plot detailed SHAP summary
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train_processed, feature_names=all_feature_names)
    plt.title('SHAP Value Distribution')
    plt.show()

# Save the best model
def save_model(model, filepath):
    """Save the trained model to disk"""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

# Main execution
def main():
    # Load data
    data_path = 'customer_churn_data.csv'  # Update with your data path
    df = load_data(data_path)
    
    # Perform EDA
    perform_eda(df)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    # Train models
    best_models = train_models(X_train, y_train, preprocessor)
    
    # Evaluate models
    results_df = evaluate_models(best_models, X_test, y_test)
    
    # Select the best model based on ROC AUC
    best_model_name = results_df.loc[results_df['ROC AUC'].idxmax(), 'Model']
    best_model = best_models[best_model_name]['model']
    
    # Analyze feature importance
    analyze_feature_importance(best_model, X_train, preprocessor)
    
    # Save the best model
    save_model(best_model, 'best_churn_model.joblib')

if __name__ == '__main__':
    main()