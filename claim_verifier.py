import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler, Nystroem
import joblib
import string
import time

class ClaimVerifier:
    def __init__(self):
        # Initialize the SVM classifier with optimized parameters
        self.classifier = None
        self.vectorizer = None
        self.pipeline = None
        self.labels = None
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove punctuation but preserve special characters like question marks and exclamation points
        # as they may indicate the tone of the claim
        text = re.sub(r'[%s]' % re.escape(string.punctuation.replace('?', '').replace('!', '')), ' ', text)
        
        # Replace question marks and exclamation points with special tokens
        text = text.replace('?', ' QUESTIONMARK ')
        text = text.replace('!', ' EXCLAMATIONMARK ')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        
        # Remove some stop words but keep negation words as they change the meaning
        negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor'}
        filtered_stop_words = stop_words - negation_words
        
        # Use simple split instead of word_tokenize to avoid the punkt_tab error
        word_tokens = text.split()
        filtered_text = [word for word in word_tokens if word not in filtered_stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        stemmed_text = [stemmer.stem(word) for word in filtered_text]
        
        return ' '.join(stemmed_text)
    
    def extract_additional_features(self, df):
        # Extract additional features that may help in claim verification
        
        # Length of statement (word count)
        df['word_count'] = df['statement'].apply(lambda x: len(str(x).split()))
        
        # Character count
        df['char_count'] = df['statement'].apply(len)
        
        # Question mark count - questions might be less factual
        df['question_count'] = df['statement'].apply(lambda x: x.count('?'))
        
        # Exclamation mark count - emotional claims might be less factual
        df['exclamation_count'] = df['statement'].apply(lambda x: x.count('!'))
        
        # Number count - claims with specific numbers may be more factual
        df['number_count'] = df['statement'].apply(lambda x: sum(c.isdigit() for c in x))
        
        # Capital letters percentage - ALL CAPS might indicate exaggeration
        df['capital_ratio'] = df['statement'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        # Extract TF-IDF features from the statement
        return df
    
    def load_liar_dataset(self, train_path, test_path, valid_path=None):
        # Load training data
        train_df = pd.read_csv(train_path, delimiter='\t', header=None,
                             names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state',
                                   'party', 'barely_true_counts', 'false_counts', 'half_true_counts',
                                   'mostly_true_counts', 'pants_on_fire_counts', 'context'])
        
        # Load test data
        test_df = pd.read_csv(test_path, delimiter='\t', header=None,
                            names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state',
                                  'party', 'barely_true_counts', 'false_counts', 'half_true_counts',
                                  'mostly_true_counts', 'pants_on_fire_counts', 'context'])
        
        # Load validation data if provided
        if valid_path:
            valid_df = pd.read_csv(valid_path, delimiter='\t', header=None,
                                 names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state',
                                       'party', 'barely_true_counts', 'false_counts', 'half_true_counts',
                                       'mostly_true_counts', 'pants_on_fire_counts', 'context'])
        else:
            valid_df = None
        
        # Preprocess statements
        train_df['processed_statement'] = train_df['statement'].apply(self.preprocess_text)
        test_df['processed_statement'] = test_df['statement'].apply(self.preprocess_text)
        
        if valid_df is not None:
            valid_df['processed_statement'] = valid_df['statement'].apply(self.preprocess_text)
        
        # Extract additional features
        train_df = self.extract_additional_features(train_df)
        test_df = self.extract_additional_features(test_df)
        
        if valid_df is not None:
            valid_df = self.extract_additional_features(valid_df)
        
        # Convert to binary classification
        # Group 'true' and 'mostly-true' as 'true'
        # Group 'false', 'pants-fire', 'barely-true', and 'half-true' as 'false'
        train_df['binary_label'] = train_df['label'].apply(self.binarize_label)
        test_df['binary_label'] = test_df['label'].apply(self.binarize_label)
        
        if valid_df is not None:
            valid_df['binary_label'] = valid_df['label'].apply(self.binarize_label)
        
        # Add history features based on speaker's past reliability
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
    
    def binarize_label(self, label):
        if label in ['true', 'mostly-true']:
            return 1  # True
        else:  # 'false', 'pants-fire', 'barely-true', 'half-true'
            return 0  # False
    
    def create_pipeline(self):
        # Create a pipeline with multiple stages
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.7,
            ngram_range=(1, 2),  # Include bigrams
            sublinear_tf=True
        )
        
        # Setup the pipeline
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('feature_selection', SelectKBest(chi2, k=1000)),  # Select top features
            ('svd', TruncatedSVD(n_components=100)),  # Dimensionality reduction (approximation to LSA)
            ('scaler', StandardScaler()),  # Normalize features
            ('svm', SVC(probability=True))  # The SVM classifier
        ])
    



# 'svm__C': [0.1, 1, 10, 100],
# 'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
# 'svm__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
# 'svm__degree': [2, 3, 4],  # For polynomial kernel
# 'svm__coef0': [0.0, 0.1, 0.5],  # For polynomial and sigmoid kernels
# 'feature_selection__k': [500, 1000, 2000]

    def train(self, train_df):
        # Create the pipeline if it doesn't exist
        if self.pipeline is None:
            self.create_pipeline()
        
        # Define a more focused parameter grid for grid search
        param_grid = {
            'svm__C': [1, 10],
            'svm__kernel': ['linear', 'rbf'],
            'svm__gamma': ['scale', 0.1],
            'feature_selection__k': [1000]
        }
        
        # Get text features
        X_text = train_df['processed_statement']
        
        # Get additional features
        X_additional = train_df[['word_count', 'char_count', 'question_count', 
                                 'exclamation_count', 'number_count', 'capital_ratio',
                                 'credibility_score']].values
        
        # Target
        y = train_df['binary_label']
        
        # Perform grid search with cross-validation
        print("Performing grid search for optimal parameters...")
        start_time = time.time()
        
        # Use stratified k-fold with fewer folds to handle potential class imbalance
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            self.pipeline, 
            param_grid, 
            cv=cv, 
            scoring='f1',
            verbose=1,
            n_jobs=-1
        )
        
        # Fit the grid search to the data
        grid_search.fit(X_text, y)
        
        # Print the best parameters and score
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print(f"Grid search completed in {time.time() - start_time:.2f} seconds")
        
        # Update the pipeline with the best parameters
        self.pipeline = grid_search.best_estimator_
        
        # Update classifier reference for easier access
        self.classifier = self.pipeline.named_steps['svm']
        
        # Check if the best kernel is RBF, and try kernel approximation if it is
        best_kernel = grid_search.best_params_.get('svm__kernel')
        if best_kernel == 'rbf':
            print("Best kernel is RBF. Trying kernel approximation techniques...")
            self._try_kernel_approximation(X_text, y, cv)
        
        # Perform cross-validation with the best pipeline
        scores = cross_val_score(self.pipeline, X_text, y, cv=cv, scoring='f1')
        print(f"Cross-validation F1 scores: {scores}")
        print(f"Mean CV F1 score: {np.mean(scores):.4f}")
        
        # Train the final model
        self.pipeline.fit(X_text, y)
    
    def _try_kernel_approximation(self, X_text, y, cv):
        """
        Try different kernel approximation techniques for potentially faster prediction
        without sacrificing too much accuracy
        """
        # Check if the classifier uses RBF kernel
        if not hasattr(self.classifier, 'gamma') or self.classifier.kernel != 'rbf':
            print("Kernel approximation only applicable for RBF kernels. Skipping.")
            return
            
        # Original pipeline for comparison
        original_score = np.mean(cross_val_score(self.pipeline, X_text, y, cv=cv, scoring='f1'))
        print(f"Original RBF kernel score: {original_score:.4f}")
        
        try:
            # Try RBFSampler approximation
            rbf_pipeline = Pipeline([
                ('tfidf', self.vectorizer),
                ('feature_selection', self.pipeline.named_steps['feature_selection']),
                ('svd', self.pipeline.named_steps['svd']),
                ('scaler', self.pipeline.named_steps['scaler']),
                ('rbf_sampler', RBFSampler(gamma=self.classifier.gamma, n_components=100, random_state=42)),
                ('linear_svc', LinearSVC(C=self.classifier.C))
            ])
            
            rbf_score = np.mean(cross_val_score(rbf_pipeline, X_text, y, cv=cv, scoring='f1'))
            print(f"RBFSampler approximation score: {rbf_score:.4f}")
            
            # Try Nystroem approximation
            nystroem_pipeline = Pipeline([
                ('tfidf', self.vectorizer),
                ('feature_selection', self.pipeline.named_steps['feature_selection']),
                ('svd', self.pipeline.named_steps['svd']),
                ('scaler', self.pipeline.named_steps['scaler']),
                ('nystroem', Nystroem(gamma=self.classifier.gamma, n_components=100, random_state=42)),
                ('linear_svc', LinearSVC(C=self.classifier.C))
            ])
            
            nystroem_score = np.mean(cross_val_score(nystroem_pipeline, X_text, y, cv=cv, scoring='f1'))
            print(f"Nystroem approximation score: {nystroem_score:.4f}")
            
            # Compare scores and use the best approach
            best_score = max(original_score, rbf_score, nystroem_score)
            
            if best_score == rbf_score and (rbf_score > original_score * 0.98):  # Only switch if not much worse
                print("Using RBFSampler approximation for faster prediction")
                self.pipeline = rbf_pipeline
            elif best_score == nystroem_score and (nystroem_score > original_score * 0.98):
                print("Using Nystroem approximation for faster prediction")
                self.pipeline = nystroem_pipeline
            else:
                print("Keeping original RBF kernel (best performance)")
                
        except Exception as e:
            print(f"Error during kernel approximation: {str(e)}")
            print("Keeping original kernel implementation")
    
    def evaluate(self, test_df):
        # Get text features
        X_test_text = test_df['processed_statement']
        
        # Get additional features
        X_test_additional = test_df[['word_count', 'char_count', 'question_count', 
                                     'exclamation_count', 'number_count', 'capital_ratio',
                                     'credibility_score']].values
        
        # Target
        y_test = test_df['binary_label']
        
        # Time the prediction
        start_time = time.time()
        
        # Make predictions
        predictions = self.pipeline.predict(X_test_text)
        
        prediction_time = time.time() - start_time
        print(f"Prediction time for {len(X_test_text)} samples: {prediction_time:.4f} seconds")
        print(f"Average prediction time per sample: {prediction_time/len(X_test_text)*1000:.4f} ms")
        
        # Print evaluation metrics
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        # Print accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        return predictions, accuracy
    
    def predict_claim(self, claim):
        # Preprocess the claim
        processed_claim = self.preprocess_text(claim)
        
        # Get length-based features
        word_count = len(claim.split())
        char_count = len(claim)
        question_count = claim.count('?')
        exclamation_count = claim.count('!')
        number_count = sum(c.isdigit() for c in claim)
        capital_ratio = sum(1 for c in claim if c.isupper()) / len(claim) if len(claim) > 0 else 0
        
        # Transform to TF-IDF features
        X_text = [processed_claim]
        
        # Make prediction
        if self.pipeline:
            prediction = self.pipeline.predict(X_text)[0]
            prob = self.pipeline.predict_proba(X_text)[0]
        else:
            raise ValueError("Model has not been trained or loaded yet.")
        
        # Convert numerical prediction back to label
        if prediction == 1:
            label = "true"
            confidence = prob[1]
        else:
            label = "false"
            confidence = prob[0]
        
        return label, confidence
    
    def save_model(self, model_path):
        joblib.dump(self.pipeline, model_path)
    
    def load_model(self, model_path):
        self.pipeline = joblib.load(model_path)
        # Update classifier reference for easier access
        self.classifier = self.pipeline.named_steps['svm']
        self.vectorizer = self.pipeline.named_steps['tfidf']
        
if __name__ == "__main__":
    # Initialize the claim verifier
    verifier = ClaimVerifier()
    
    # Load and preprocess the LIAR dataset
    train_df, test_df, valid_df = verifier.load_liar_dataset(
        'Liar/train.tsv',
        'Liar/test.tsv',
        'Liar/valid.tsv'
    )
    
    # Train the model
    print("Training the model...")
    verifier.train(train_df)
    
    # Evaluate the model on test set
    print("\nEvaluating the model on test set...")
    test_predictions, test_accuracy = verifier.evaluate(test_df)
    
    # Evaluate the model on validation set
    if valid_df is not None:
        print("\nEvaluating the model on validation set...")
        valid_predictions, valid_accuracy = verifier.evaluate(valid_df)
    
    # Save the trained model
    verifier.save_model('claim_verifier_model.joblib')
    
    # Example of claim verification
    test_claims = [
        "Building a wall on the U.S.-Mexico border will take literally years.",
        "The number of illegal immigrants could be 3 million. It could be 30 million.",
        "Democrats already agreed to a deal that Republicans wanted.",
        "Austin has more lobbyists working for it than any other municipality in Texas."
    ]
    
    print("\nTesting individual claims:")
    for claim in test_claims:
        verdict, confidence = verifier.predict_claim(claim)
        print(f"\nClaim: {claim}")
        print(f"Verdict: {verdict} (confidence: {confidence:.4f})") 