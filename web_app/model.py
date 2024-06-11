import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def process_file(file_path):
    # Load the dataset
    data = pd.read_excel(file_path)

    # Convert date columns to datetime
    date_columns = ['CLM_LOSS_DT', 'CLM_INTM_DT', 'POLICY_START_DATE', 'POLICY_END_DATE', 'CE_CR_DT', 'CE_DT', 'CLM_CR_DT']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')

    # Define fraud detection rules
    def detect_fraud(row):
        if row['APPROVAL_STATUS'] == 'I' and pd.notnull(row['CLM_NO']):
            return 'Fraudulent', 'Unapproved policy claim'
        if pd.notnull(row['CLM_LOSS_DT']) and pd.notnull(row['CLM_INTM_DT']) and (row['CLM_INTM_DT'] - row['CLM_LOSS_DT']).days > 30:
            return 'Further Investigation Required', 'Claim intimation delay'
        if pd.notnull(row['CLM_LOSS_DT']) and (row['CLM_LOSS_DT'] < row['POLICY_START_DATE'] or row['CLM_LOSS_DT'] > row['POLICY_END_DATE']):
            return 'Fraudulent', 'Claim outside policy period'
        if row['CE_AMT_LC_1'] == 0:
            return 'Fraudulent', 'Zero estimate amount'
        if row['CE_AMT_FC'] == row['CE_AMT_LC_1'] and row['CE_CURR_CODE'] != 'UGX':
            return 'Further Investigation Required', 'Currency discrepancy'
        if pd.notnull(row['CE_CR_DT']) and pd.notnull(row['CE_DT']) and (row['CE_DT'] - row['CE_CR_DT']).days > 30:
            return 'Further Investigation Required', 'Reserve amount delay'
        if pd.notnull(row['CLM_CR_DT']) and pd.notnull(row['POLICY_START_DATE']) and (row['CLM_CR_DT'] - row['POLICY_START_DATE']).days <= 10:
            return 'Further Investigation Required', 'Early claim after policy start'
        return 'Okay', 'No issues'

    # Apply fraud detection rules
    data['Status'], data['Reason'] = zip(*data.apply(detect_fraud, axis=1))

    # Mapping Status to a binary classification for machine learning
    data['Fraudulent'] = data['Status'].apply(lambda x: 1 if x in ['Fraudulent', 'Further Investigation Required'] else 0)

    # Selecting and preprocessing features for the model
    feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    features = data[feature_columns].fillna(data[feature_columns].median())

    # Normalize the feature data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, data['Fraudulent'], test_size=0.2, random_state=42)

    # Build and train the hybrid model (SVM + Random Forest)
    svm = SVC(probability=True)
    rf = RandomForestClassifier()
    svm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Ensemble prediction by averaging the probabilities
    svm_probs = svm.predict_proba(X_test)[:, 1]
    rf_probs = rf.predict_proba(X_test)[:, 1]
    avg_probs = (svm_probs + rf_probs) / 2
    predictions = np.round(avg_probs)

    # Evaluate the model
    report = classification_report(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)

    # Ensure static/uploads directory exists
    static_dir = os.path.join('static', 'uploads')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Save confusion matrix plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    confusion_matrix_path = os.path.join(static_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Save feature importance plot
    feature_importances = rf.feature_importances_
    importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance for Fraud Detection')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    feature_importance_path = os.path.join(static_dir, 'feature_importance.png')
    plt.savefig(feature_importance_path)
    plt.close()

    # Save distribution plot
    plt.figure(figsize=(10, 7))
    sns.countplot(x='Fraudulent', data=data)
    plt.title('Distribution of Fraudulent vs. Non-Fraudulent Claims')
    plt.xlabel('Claim Status (0: Non-Fraudulent, 1: Fraudulent)')
    plt.ylabel('Count')
    distribution_path = os.path.join(static_dir, 'distribution.png')
    plt.savefig(distribution_path)
    plt.close()

    # Output the results to an Excel file
    output_data = data[['POLICY_NUMBER', 'ASSURED_NAME', 'CLM_NO', 'Status', 'Reason']]
    output_file_path = os.path.join('uploads', 'Fraudulent_Claims_Detection_Output.xlsx')
    output_data.to_excel(output_file_path, index=False)

    return output_data, output_file_path, report, 'uploads/confusion_matrix.png', 'uploads/feature_importance.png', 'uploads/distribution.png'
