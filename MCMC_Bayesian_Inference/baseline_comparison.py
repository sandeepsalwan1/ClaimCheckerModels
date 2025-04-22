import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, auc
import joblib
import pickle
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

def preprocess_text(text):
    """Text preprocessing"""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation.replace('?', '').replace('!', '')), ' ', text)
    text = text.replace('?', ' QUESTIONMARK ')
    text = text.replace('!', ' EXCLAMATIONMARK ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # stop words
    stop_words = set(stopwords.words('english'))
    negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor'}
    filtered_stop_words = stop_words - negation_words
    word_tokens = text.split()
    filtered_text = [word for word in word_tokens if word not in filtered_stop_words]
    
    # stemming
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in filtered_text]
    
    return ' '.join(stemmed_text)

def binarize_label(label):
    """Convert multi-class labels to binary"""
    if label in ['true', 'mostly-true', 'half-true']:
        return 1  # True
    else:  # 'false', 'pants-fire', 'barely-true'
        return 0  # False

def extract_additional_features(df):
    """Extract additional features from statements"""
    df['word_count'] = df['statement'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['statement'].apply(len)
    df['question_count'] = df['statement'].apply(lambda x: x.count('?'))
    df['exclamation_count'] = df['statement'].apply(lambda x: x.count('!'))
    df['number_count'] = df['statement'].apply(lambda x: sum(c.isdigit() for c in x))
    df['capital_ratio'] = df['statement'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    return df

def load_liar_dataset(train_path, test_path, valid_path=None):
    """Load and preprocess the LIAR dataset"""
    train_df = pd.read_csv(train_path, delimiter='\t', header=None,
                         names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state',
                               'party', 'barely_true_counts', 'false_counts', 'half_true_counts',
                               'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    test_df = pd.read_csv(test_path, delimiter='\t', header=None,
                        names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state',
                              'party', 'barely_true_counts', 'false_counts', 'half_true_counts',
                              'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    if valid_path:
        valid_df = pd.read_csv(valid_path, delimiter='\t', header=None,
                             names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state',
                                   'party', 'barely_true_counts', 'false_counts', 'half_true_counts',
                                   'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    else:
        valid_df = None
    
    # Preprocess statements
    train_df['processed_statement'] = train_df['statement'].apply(preprocess_text)
    test_df['processed_statement'] = test_df['statement'].apply(preprocess_text)
    
    if valid_df is not None:
        valid_df['processed_statement'] = valid_df['statement'].apply(preprocess_text)
    
    # Extract additional features
    train_df = extract_additional_features(train_df)
    test_df = extract_additional_features(test_df)
    
    if valid_df is not None:
        valid_df = extract_additional_features(valid_df)
    
    # Convert to binary classification
    train_df['binary_label'] = train_df['label'].apply(binarize_label)
    test_df['binary_label'] = test_df['label'].apply(binarize_label)
    
    if valid_df is not None:
        valid_df['binary_label'] = valid_df['label'].apply(binarize_label)
    
    # Add credibility score feature
    train_df['credibility_score'] = (
        train_df['mostly_true_counts'] + train_df['half_true_counts'] - 
        train_df['false_counts'] - 2*train_df['pants_on_fire_counts']
    ) / (train_df['barely_true_counts'] + train_df['false_counts'] + 
         train_df['half_true_counts'] + train_df['mostly_true_counts'] + 
         train_df['pants_on_fire_counts'] + 1)
    
    test_df['credibility_score'] = (
        test_df['mostly_true_counts'] + test_df['half_true_counts'] - 
        test_df['false_counts'] - 2*test_df['pants_on_fire_counts']
    ) / (test_df['barely_true_counts'] + test_df['false_counts'] + 
         test_df['half_true_counts'] + test_df['mostly_true_counts'] + 
         test_df['pants_on_fire_counts'] + 1)
    
    if valid_df is not None:
        valid_df['credibility_score'] = (
            valid_df['mostly_true_counts'] + valid_df['half_true_counts'] - 
            valid_df['false_counts'] - 2*valid_df['pants_on_fire_counts']
        ) / (valid_df['barely_true_counts'] + valid_df['false_counts'] + 
             valid_df['half_true_counts'] + valid_df['mostly_true_counts'] + 
             valid_df['pants_on_fire_counts'] + 1)
    
    return train_df, test_df, valid_df

class BaselineLogisticRegression:
    def __init__(self):
        """Initialize the baseline model with TF-IDF + Logistic Regression"""
        self.pipeline = None
        
    def create_pipeline(self):
        """Create a TF-IDF + Logistic Regression pipeline"""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7, ngram_range=(1, 2))),
            ('logreg', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='liblinear'))
        ])
        
    def train(self, train_df):
        """Train the baseline model"""
        if self.pipeline is None:
            self.create_pipeline()
        
        X_text = train_df['processed_statement']
        y = train_df['binary_label']
        
        print("Training the baseline TF-IDF + Logistic Regression model...")
        start_time = time.time()
        
        self.pipeline.fit(X_text, y)
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        
    def evaluate(self, test_df):
        """Evaluate the baseline model"""
        X_test_text = test_df['processed_statement']
        y_test = test_df['binary_label']
        
        start_time = time.time()
        
        predictions = self.pipeline.predict(X_test_text)
        prediction_probs = self.pipeline.predict_proba(X_test_text)[:, 1]
        
        prediction_time = time.time() - start_time
        print(f"Prediction time for {len(X_test_text)} samples: {prediction_time:.4f} seconds")
        print(f"Average prediction time per sample: {prediction_time/len(X_test_text)*1000:.4f} ms")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='binary'
        )
        
        print("\nBaseline Classification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nBaseline Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        print(f"\nBaseline Accuracy: {accuracy:.4f}")
        print(f"Baseline Precision: {precision:.4f}")
        print(f"Baseline Recall: {recall:.4f}")
        print(f"Baseline F1 Score: {f1:.4f}")
        
        fpr, tpr, _ = roc_curve(y_test, prediction_probs)
        roc_auc = auc(fpr, tpr)
        print(f"Baseline ROC AUC: {roc_auc:.4f}")
        
        # ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - Baseline Model')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('baseline_roc_curve.png', dpi=300)
        
        # comparison metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': predictions,
            'prediction_probs': prediction_probs
        }
    
    def save_model(self, model_path):
        """Save the baseline model"""
        joblib.dump(self.pipeline, model_path)
    
    def load_model(self, model_path):
        """Load the baseline model"""
        self.pipeline = joblib.load(model_path)

def compare_with_mcmc(baseline_metrics, mcmc_model_path='bayesian_model.pkl'):
    """Compare baseline with MCMC model"""
    with open(mcmc_model_path, 'rb') as f:
        mcmc_model = pickle.load(f)
    
    # Load test data (assuming the same data structure as in training)
    _, test_df, _ = load_liar_dataset('liar/train.tsv', 'liar/test.tsv', 'liar/valid.tsv')
    
    X_test_text = test_df['processed_statement']
    y_test = test_df['binary_label']
    
    # Make predictions with MCMC
    X_test = np.load('liar/test_embeddings.npy')
    y_test = np.load('liar/test_labels.npy')
    y_test = np.where(y_test > 2, 1, 0)

    mcmc_predictions = mcmc_model.predict(X_test)
    mcmc_probs = mcmc_model.predict_probs(X_test)[:, 3:].sum(axis=1)
    mcmc_predictions = (mcmc_probs >= 0.5).astype(int)

    
    # MCMC metrics
    mcmc_accuracy = accuracy_score(y_test, mcmc_predictions)
    mcmc_precision, mcmc_recall, mcmc_f1, _ = precision_recall_fscore_support(
        y_test, mcmc_predictions, average='binary'
    )
    
    mcmc_fpr, mcmc_tpr, _ = roc_curve(y_test, mcmc_probs)
    mcmc_roc_auc = auc(mcmc_fpr, mcmc_tpr)

    print("\nMCMC Classification Report:")
    print(classification_report(y_test, mcmc_predictions))
    
    print("\nmcmc Confusion Matrix:")
    print(confusion_matrix(y_test, mcmc_predictions))
    
    print(f"\nMCMC Accuracy: {mcmc_accuracy:.4f}")
    print(f"MCMC Precision: {mcmc_precision:.4f}")
    print(f"MCMC Recall: {mcmc_recall:.4f}")
    print(f"MCMC F1 Score: {mcmc_f1:.4f}")
    print(f"MCMC ROC AUC: {mcmc_roc_auc:.4f}")

    accuracy_improvement = mcmc_accuracy - baseline_metrics['accuracy']
    precision_improvement = mcmc_precision - baseline_metrics['precision']
    recall_improvement = mcmc_recall - baseline_metrics['recall']
    f1_improvement = mcmc_f1 - baseline_metrics['f1']
    auc_improvement = mcmc_roc_auc - baseline_metrics['roc_auc']
    
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Baseline (TF-IDF + LogReg)': [
            baseline_metrics['accuracy'], 
            baseline_metrics['precision'], 
            baseline_metrics['recall'], 
            baseline_metrics['f1'],
            baseline_metrics['roc_auc']
        ],
        'mcmc': [mcmc_accuracy, mcmc_precision, mcmc_recall, mcmc_f1, mcmc_roc_auc],
        'Improvement': [accuracy_improvement, precision_improvement, recall_improvement, f1_improvement, auc_improvement]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['Improvement'] = comparison_df['Improvement'].apply(
        lambda x: f"{x*100:.2f}%" if x >= 0 else f"{x*100:.2f}%"
    )
    
    print("\nModel Comparison (mcmc vs. Baseline):")
    print(comparison_df.to_string(index=False))
    
    # bar chart
    plt.figure(figsize=(12, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    baseline_values = [
        baseline_metrics['accuracy'], 
        baseline_metrics['precision'], 
        baseline_metrics['recall'], 
        baseline_metrics['f1'],
        baseline_metrics['roc_auc']
    ]
    mcmc_values = [mcmc_accuracy, mcmc_precision, mcmc_recall, mcmc_f1, mcmc_roc_auc]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='Baseline (TF-IDF + LogReg)')
    plt.bar(x + width/2, mcmc_values, width, label='MCMC')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison: Baseline vs. MCMC')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, v in enumerate(baseline_values):
        plt.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    for i, v in enumerate(mcmc_values):
        plt.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()
    
    # Plot both ROC curves for comparison
    plt.figure(figsize=(10, 8))
    plt.plot(mcmc_fpr, mcmc_tpr, color='darkorange', lw=2, label=f'mcmc ROC (AUC = {mcmc_roc_auc:.3f})')
    plt.plot(baseline_metrics['fpr'], baseline_metrics['tpr'], color='blue', lw=2, 
             label=f'Baseline ROC (AUC = {baseline_metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison: mcmc vs. Baseline')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_comparison.png', dpi=300)
    plt.show()
    
    return comparison_df

if __name__ == "__main__":
    baseline = BaselineLogisticRegression()
    
    train_df, test_df, valid_df = load_liar_dataset(
        'Liar/train.tsv',
        'Liar/test.tsv',
        'Liar/valid.tsv'
    )
    
    baseline.train(train_df)
    baseline_metrics = baseline.evaluate(test_df)
    baseline.save_model('baseline_model.joblib')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(test_df['binary_label'], baseline_metrics['prediction_probs'])
    baseline_metrics['fpr'] = fpr
    baseline_metrics['tpr'] = tpr
    
    # compare with MCMC
    compare_with_mcmc(baseline_metrics)