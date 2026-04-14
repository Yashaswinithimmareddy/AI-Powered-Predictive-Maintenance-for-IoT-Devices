import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------------------------------------
# Phase 1: Setup & Directories
# ---------------------------------------------------------
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('images', exist_ok=True)

# ---------------------------------------------------------
# Phase 2: Virtual Simulation (Dataset Generation)
# ---------------------------------------------------------
def generate_simulated_dataset(path='data/sensor_data.csv'):
    """
    Simulates real IoT sensor data from industrial machines.
    Generates Temperature, Vibration, Pressure, and RPM readings.
    """
    print("[INFO] Simulating IoT sensor data...")
    np.random.seed(42)  # For reproducible results
    n_samples = 5000
    
    # Simulate normal operating parameters
    temperature = np.random.normal(loc=70, scale=5, size=n_samples) 
    vibration = np.random.normal(loc=5, scale=1.5, size=n_samples)
    pressure = np.random.normal(loc=100, scale=10, size=n_samples)
    rpm = np.random.normal(loc=2000, scale=50, size=n_samples)
    
    # Introduce anomalies leading to machine failure
    failure = np.zeros(n_samples)
    
    for i in range(n_samples):
        # 15% of the time, the machine experiences abnormal conditions
        if np.random.rand() > 0.85: 
            temperature[i] += np.random.normal(15, 3) # Overheating
            vibration[i] += np.random.normal(4, 1)    # Severe shaking
            
        # Failure rule: High temp AND high vibration usually causes failure
        if temperature[i] > 82 and vibration[i] > 7.5:
            if np.random.rand() > 0.2: # 80% chance of failure under these conditions
                failure[i] = 1
                
    df = pd.DataFrame({
        'Machine_ID': [f'MCH-{i:04d}' for i in range(n_samples)],
        'Temperature_C': temperature,
        'Vibration_mm_s': vibration,
        'Pressure_psi': pressure,
        'Rotational_Speed_rpm': rpm,
        'Failure': failure.astype(int)
    })
    
    df.to_csv(path, index=False)
    print(f"[SUCCESS] Simulated dataset saved to '{path}' ({n_samples} records).")
    return df

# ---------------------------------------------------------
# Phase 3 & 4: Data Cleaning and Feature Engineering
# ---------------------------------------------------------
def load_and_preprocess(filepath):
    """
    Loads the dataset and prepares it for the ML model.
    """
    print("\n[INFO] Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    
    # Feature Engineering: We don't need 'Machine_ID' to predict failure
    X = df.drop(columns=['Machine_ID', 'Failure'])
    y = df['Failure']  # The target variable (0 = Healthy, 1 = Failure)
    
    # Split into 80% training data and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Training records: {X_train.shape[0]}, Testing records: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------
# Phase 5: Model Building
# ---------------------------------------------------------
def train_model(X_train, y_train):
    """
    Trains a Random Forest classification model.
    """
    print("\n[INFO] Training Random Forest AI Model...")
    # class_weight='balanced' helps handles the fact that failures are rare
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("[SUCCESS] Model trained successfully.")
    return model

# ---------------------------------------------------------
# Phase 6 & 8: Evaluation & Visualization
# ---------------------------------------------------------
def evaluate_and_visualize(model, X_test, y_test):
    """
    Evaluates the model and generates visual proof of performance.
    """
    print("\n[INFO] Evaluating Model Performance...")
    y_pred = model.predict(X_test)
    
    # 1. Text Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Assessment ---")
    print(f"Prediction Accuracy: {acc * 100:.2f}%\n")
    print("Detailed Classification Report (Precision & Recall):")
    print(classification_report(y_test, y_pred))
    
    # 2. Save Prediction Results to CSV for Proof
    results_df = X_test.copy()
    results_df['Actual_Status'] = y_test
    results_df['Predicted_Status'] = y_pred
    results_df.to_csv('outputs/prediction_results.csv', index=False)
    print("[SUCCESS] Saved sample predictions to 'outputs/prediction_results.csv'.")
    
    # 3. Visualization: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy (0)', 'Failure (1)'], 
                yticklabels=['Healthy (0)', 'Failure (1)'])
    plt.title('AI Prediction Accuracy (Confusion Matrix)')
    plt.ylabel('Actual Machine Condition')
    plt.xlabel('AI Predicted Condition')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png', dpi=300)
    plt.close()
    print("[SUCCESS] Saved confusion matrix visual to 'images/confusion_matrix.png'.")
    
    # 4. Visualization: Feature Importance (Which sensor matters most?)
    importances = model.feature_importances_
    features = X_test.columns
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=features, hue=features, palette='viridis', legend=False)
    plt.title('Sensor Importance in Predicting Machine Failures')
    plt.xlabel('Impact Score')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png', dpi=300)
    plt.close()
    print("[SUCCESS] Saved feature importance visual to 'images/feature_importance.png'.")

# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
def main():
    print("==================================================================")
    print("   AI-POWERED PREDICTIVE MAINTENANCE SYSTEM FOR IoT DEVICES ")
    print("==================================================================\n")
    
    data_path = 'data/sensor_data.csv'
    
    # Step 1: Simulate the Data
    generate_simulated_dataset(data_path)
    
    # Step 2: Prepare the Data
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    
    # Step 3: Train the Machine Learning Model
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate and Predict
    evaluate_and_visualize(model, X_test, y_test)
    
    print("\n==================================================================")
    print(" SYSTEM RUN COMPLETE. Check 'images' and 'outputs' for results. ")
    print("==================================================================")

if __name__ == '__main__':
    main()
