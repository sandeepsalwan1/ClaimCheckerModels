import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve
import pandas as pd
import matplotlib as mpl

class SVMVisualizer:
    """
    A class to visualize the performance of an SVM classifier.
    
    This class creates several plots to help analyze and understand
    the performance of a Support Vector Machine (SVM) model:
    1. Confusion Matrix (with default and optimal thresholds)
    2. F1 Score vs Threshold
    3. Probability Distribution
    4. ROC Curve
    """
    
    def __init__(self, y_true, y_pred, y_prob, output_dir='.'):
        """
        Initialize the visualizer with the model's predictions.
        
        Args:
            y_true: Array-like of true labels (0 for false, 1 for true)
            y_pred: Array-like of predicted labels (0 for false, 1 for true)
            y_prob: Array-like of predicted probabilities for the positive class (class 1)
            output_dir: Directory to save the generated plots
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob)
        self.output_dir = output_dir
        self.optimal_threshold = None
        
        # Set style for all plots - use a clean, modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['axes.titlesize'] = 14
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['figure.titlesize'] = 16
        
        # Find the optimal threshold first as we'll need it in multiple plots
        self._find_optimal_threshold()
    
    def _find_optimal_threshold(self):
        """Find the threshold that maximizes F1 score"""
        thresholds = np.arange(0, 1.01, 0.01)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_prob >= threshold).astype(int)
            f1 = f1_score(self.y_true, y_pred_thresh)
            f1_scores.append(f1)
        
        # Find the threshold that maximizes F1
        best_threshold_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[best_threshold_idx]
        self.best_f1 = f1_scores[best_threshold_idx]
        self.f1_thresholds = thresholds
        self.f1_scores = f1_scores
    
    def plot_confusion_matrix(self, save=True):
        """
        Plot confusion matrices for both default (0.5) and optimal thresholds.
        """
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Get predictions with default threshold (0.5)
        default_preds = (self.y_prob >= 0.5).astype(int)
        default_cm = confusion_matrix(self.y_true, default_preds)
        
        # Get predictions with optimal threshold
        optimal_preds = (self.y_prob >= self.optimal_threshold).astype(int)
        optimal_cm = confusion_matrix(self.y_true, optimal_preds)
        
        # First confusion matrix (default threshold)
        sns.heatmap(default_cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["False", "True"], 
                   yticklabels=["False", "True"],
                   ax=ax1, annot_kws={"size": 16})
        
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_title(f'Confusion Matrix (Default Threshold = 0.5)', fontsize=14)
        
        # Second confusion matrix (optimal threshold)
        sns.heatmap(optimal_cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["False", "True"], 
                   yticklabels=["False", "True"],
                   ax=ax2, annot_kws={"size": 16})
        
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_title(f'Confusion Matrix (Optimal Threshold = {self.optimal_threshold:.2f})', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_f1_threshold(self, save=True):
        """
        Plot F1 score as a function of the classification threshold.
        Identify the threshold that maximizes the F1 score.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot F1 vs threshold
        plt.plot(self.f1_thresholds, self.f1_scores, lw=2, color='#1f77b4')
        
        # Mark the optimal threshold
        plt.axvline(x=self.optimal_threshold, color='red', linestyle='--', alpha=0.7)
        
        # Add some text describing the optimal threshold
        plt.text(self.optimal_threshold + 0.05, self.best_f1 * 0.97, 
                f'Optimal Threshold = {self.optimal_threshold:.2f}',
                fontsize=12, color='red')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('F1 Score vs Threshold', fontsize=14)
        
        if save:
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/f1_threshold.png", dpi=300, bbox_inches='tight')
            
        plt.show()
        
        return self.optimal_threshold, self.best_f1
        
    def plot_probability_distribution(self, save=True):
        """
        Plot histograms of the predicted probability distributions
        for both the positive and negative classes.
        """
        plt.figure(figsize=(12, 8))
        
        # Split probabilities by true class
        prob_false_class = self.y_prob[self.y_true == 0]
        prob_true_class = self.y_prob[self.y_true == 1]
        
        # Plot histograms using more sophisticated styling
        plt.hist(prob_false_class, bins=25, alpha=0.7, color='#6baed6', 
                label='False Claims', density=False)
        plt.hist(prob_true_class, bins=25, alpha=0.7, color='#fd8d3c', 
                label='True Claims', density=False)
        
        # Add vertical line at optimal threshold
        plt.axvline(x=self.optimal_threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Optimal Threshold = {self.optimal_threshold:.2f}')
        
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Probability Distribution by Class', fontsize=14)
        plt.legend(loc='upper right', frameon=True)
        plt.grid(alpha=0.3)
        
        if save:
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/probability_distribution.png", dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_roc_curve(self, save=True):
        """
        Plot the Receiver Operating Characteristic (ROC) curve
        and calculate the Area Under the Curve (AUC).
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate ROC curve and ROC area
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color='#1f77b4', lw=2,
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save:
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
            
        plt.show()
        
        return roc_auc
    
    def generate_all_plots(self):
        """Generate all the visualization plots at once."""
        self.plot_confusion_matrix()
        self.plot_f1_threshold()
        self.plot_probability_distribution()
        roc_auc = self.plot_roc_curve()
        
        # Return a summary of metrics
        return {
            "best_threshold": self.optimal_threshold,
            "best_f1_score": self.best_f1,
            "roc_auc": roc_auc
        }


def plot_claim_verifier_results(test_df, predictions, probs, output_dir='.'):
    """
    Helper function to visualize results from the ClaimVerifier.
    
    Args:
        test_df: Pandas DataFrame containing the test data
        predictions: Array-like of predicted labels
        probs: Array-like of predicted probabilities (for the positive class)
        output_dir: Directory to save the plots
    """
    # Get true labels
    y_true = test_df['binary_label'].values
    
    # Create the visualizer and generate all plots
    visualizer = SVMVisualizer(y_true, predictions, probs, output_dir)
    metrics = visualizer.generate_all_plots()
    
    print("\nVisualization Summary:")
    print(f"Best Classification Threshold: {metrics['best_threshold']:.3f}")
    print(f"Best F1 Score: {metrics['best_f1_score']:.3f}")
    print(f"ROC AUC Score: {metrics['roc_auc']:.3f}")
    
    # Calculate accuracy at different thresholds
    default_preds = (probs >= 0.5).astype(int)
    optimal_preds = (probs >= metrics['best_threshold']).astype(int)
    
    default_accuracy = np.mean(default_preds == y_true)
    optimal_accuracy = np.mean(optimal_preds == y_true)
    
    print(f"Default Threshold (0.5) Accuracy: {default_accuracy:.3f}")
    print(f"Optimal Threshold ({metrics['best_threshold']:.3f}) Accuracy: {optimal_accuracy:.3f}")
    
    return metrics


if __name__ == "__main__":
    # Example usage:
    # If this script is run directly, it can load the model and test data
    # to generate the visualizations
    
    import os
    from claim_verifier import ClaimVerifier
    
    # Initialize the claim verifier
    verifier = ClaimVerifier()
    
    # Check if a model exists
    model_path = 'claim_verifier_model.joblib'
    if os.path.exists(model_path):
        # Load the model
        verifier.load_model(model_path)
        
        # Load test data
        train_df, test_df, valid_df = verifier.load_liar_dataset(
            'Liar/train.tsv',
            'Liar/test.tsv',
            'Liar/valid.tsv'
        )
        
        # Get predictions and probabilities
        X_test_text = test_df['processed_statement']
        y_test = test_df['binary_label'].values
        
        # Get predictions
        predictions = verifier.pipeline.predict(X_test_text)
        
        # Get probabilities (for the positive class)
        probabilities = verifier.pipeline.predict_proba(X_test_text)[:, 1]
        
        # Create output directory if it doesn't exist
        output_dir = 'visualizations'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save visualizations
        plot_claim_verifier_results(test_df, predictions, probabilities, output_dir)
        
        print(f"\nAll visualizations have been saved to the '{output_dir}' directory.")
    else:
        print(f"Model file '{model_path}' not found. Please train the model first.") 