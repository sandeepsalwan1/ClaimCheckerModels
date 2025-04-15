import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import pandas as pd
import os
import joblib
from claim_verifier import ClaimVerifier
import matplotlib as mpl

# Import the visualizer class
from svm_visualization import SVMVisualizer

def visualize_saved_model(model_path='claim_verifier_model.joblib', 
                          test_data_path='Liar/test.tsv',
                          output_dir='visualizations'):
    """
    Visualize the performance of a saved SVM model without retraining.
    
    Args:
        model_path: Path to the saved joblib model file
        test_data_path: Path to the test data TSV file
        output_dir: Directory to save the generated plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure matplotlib for better appearance
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    print(f"Loading model from {model_path}...")
    
    # Load the model from joblib file
    model = joblib.load(model_path)
    
    # Initialize the claim verifier just for data loading functionality
    verifier = ClaimVerifier()
    
    # We'll load just the test data
    print(f"Loading test data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path, delimiter='\t', header=None,
                      names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state',
                            'party', 'barely_true_counts', 'false_counts', 'half_true_counts',
                            'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    
    # Preprocess the statements
    print("Preprocessing test data...")
    test_df['processed_statement'] = test_df['statement'].apply(verifier.preprocess_text)
    
    # Extract additional features
    test_df = verifier.extract_additional_features(test_df)
    
    # Convert to binary classification
    test_df['binary_label'] = test_df['label'].apply(verifier.binarize_label)
    
    # Add speaker history features
    # test_df['speaker_history'] = test_df['speaker'].apply(verifier.get_speaker_history)
    test_df['credibility_score'] = (
        test_df['mostly_true_counts'] + test_df['half_true_counts'] - 
        test_df['false_counts'] - 2*test_df['pants_on_fire_counts']
    ) / (test_df['barely_true_counts'] + test_df['false_counts'] + 
         test_df['half_true_counts'] + test_df['mostly_true_counts'] + 
         test_df['pants_on_fire_counts'] + 1)
    
    # Add party features
    test_df['party_republican'] = test_df['party'].apply(lambda x: 1 if 'republican' in str(x).lower() else 0)
    test_df['party_democrat'] = test_df['party'].apply(lambda x: 1 if 'democrat' in str(x).lower() else 0)
    
    # Get text features for prediction
    X_test_text = test_df['processed_statement']
    
    # Get true labels
    y_test = test_df['binary_label'].values
    
    # Time the prediction
    print("Generating predictions...")
    import time
    start_time = time.time()
    
    # Make predictions
    predictions = model.predict(X_test_text)
    
    # Get probabilities for the positive class (class 1)
    if hasattr(model, 'predict_proba'):
        # For models that provide probabilities directly
        probabilities = model.predict_proba(X_test_text)[:, 1]
    else:
        # For models like LinearSVC that provide decision function scores
        # We'll transform these to pseudo-probabilities
        decision_scores = model.decision_function(X_test_text)
        # Simple scaling to [0,1] range (not true probabilities but works for visualization)
        probabilities = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
    
    prediction_time = time.time() - start_time
    print(f"Prediction time for {len(X_test_text)} samples: {prediction_time:.4f} seconds")
    print(f"Average prediction time per sample: {prediction_time/len(X_test_text)*1000:.4f} ms")
    
    # Create the visualizer and generate the plots
    print("\nGenerating visualizations...")
    visualizer = SVMVisualizer(y_test, predictions, probabilities, output_dir)
    
    # Generate all plots
    metrics = visualizer.generate_all_plots()
    auc = metrics['roc_auc']
    
    # Find optimal threshold predictions
    optimal_preds = (probabilities >= metrics['best_threshold']).astype(int)
    default_preds = (probabilities >= 0.5).astype(int)
    
    # Calculate accuracy scores
    default_accuracy = np.mean(default_preds == y_test)
    optimal_accuracy = np.mean(optimal_preds == y_test)
    
    # Print summary
    print("\n========= MODEL PERFORMANCE SUMMARY =========")
    print(f"Default Threshold (0.5) Accuracy: {default_accuracy:.4f}")
    print(f"Optimal Threshold ({metrics['best_threshold']:.3f}) Accuracy: {optimal_accuracy:.4f}")
    print(f"Best F1 Score: {metrics['best_f1_score']:.4f}")
    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
    print("============================================")
    
    print(f"\nAll visualizations have been saved to the '{output_dir}' directory.")
    
    return metrics

if __name__ == "__main__":
    # Check if the model file exists
    model_path = 'claim_verifier_model.joblib'
    if os.path.exists(model_path):
        visualize_saved_model(model_path)
    else:
        print(f"Model file '{model_path}' not found. Please train the model first.")
        print("You can train the model by running: python claim_verifier.py") 