import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import string
import time
import os
from datetime import datetime
import warnings
import spacy
from tqdm import tqdm
import torch

# Import CUDA support if available
try:
    import cuml
    from cuml.svm import SVC as cuSVC
    from cuml.decomposition import TruncatedSVD as cuTruncatedSVD
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    HAS_CUDA = True
    
    # Check if CUDA is available in PyTorch as well
    PYTORCH_CUDA = torch.cuda.is_available()
    if PYTORCH_CUDA:
        DEVICE = torch.device("cuda")
        CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
        print(f"GPU acceleration enabled with RAPIDS cuML and PyTorch on {CUDA_DEVICE_NAME}")
    else:
        print("GPU acceleration enabled with RAPIDS cuML")
except ImportError:
    HAS_CUDA = False
    PYTORCH_CUDA = torch.cuda.is_available()
    if PYTORCH_CUDA:
        DEVICE = torch.device("cuda")
        CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
        print(f"PyTorch GPU acceleration available on {CUDA_DEVICE_NAME}")
        print("RAPIDS cuML not found. For full GPU acceleration, install: pip install cuml-cu12")
    else:
        print("No GPU acceleration available. Using CPU version.")

# Filter specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The default value of `n_init`")
os.environ["PYTHONWARNINGS"] = "ignore"

# Try to load spaCy model for better NLP features
try:
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
    print("Using spaCy for advanced NLP features")
except:
    HAS_SPACY = False
    print("spaCy model not found. Using basic NLP. For better results: python -m spacy download en_core_web_sm")

# Configure PyTorch
if PYTORCH_CUDA:
    # Set optimal PyTorch settings for RTX 4090
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    
    # Print GPU memory info
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

class AdvancedTextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Enhanced transformer to extract advanced linguistic features from text"""
    
    def __init__(self):
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            print("Note: NLTK resources couldn't be downloaded, will use basic features only")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, list):
            X = pd.Series(X)
        elif not isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.Series(X)
        features = pd.DataFrame()
        
        # Lexical diversity ratio (unique words / total words)
        features['lexical_diversity'] = X.apply(
            lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1)
        )
        
        # Average word length (longer words may indicate more formal, truthful claims)
        features['avg_word_length'] = X.apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Count hedging words (may indicate uncertainty)
        hedging_words = ['may', 'might', 'could', 'perhaps', 'possibly', 'supposedly',
                        'allegedly', 'apparently', 'seemingly', 'reportedly', 'rumor',
                        'rumors', 'sort of', 'kind of', 'mostly', 'usually', 'mainly']
        features['hedging_ratio'] = X.apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in hedging_words) / 
                    (len(str(x).split()) + 1)
        )
        
        # Count certainty words (may indicate truthfulness)
        certainty_words = ['definitely', 'certainly', 'absolutely', 'undoubtedly', 'clearly',
                         'obviously', 'indeed', 'truly', 'actually', 'fact', 'facts', 'proven',
                         'evidence', 'study', 'research', 'confirmed', 'official', 'verified']
        features['certainty_ratio'] = X.apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in certainty_words) / 
                     (len(str(x).split()) + 1)
        )
        
        # Count past tense verbs (research shows deceptive statements use less past tense)
        past_tense_verbs = ['was', 'were', 'had', 'did', 'said', 'went', 'came', 'took', 'made',
                           'knew', 'thought', 'got', 'told', 'found', 'felt', 'showed']
        features['past_tense_ratio'] = X.apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in past_tense_verbs) / 
                     (len(str(x).split()) + 1)
        )
        
        # First-person pronoun ratio (liars use less first-person pronouns)
        first_person = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
        features['first_person_ratio'] = X.apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in first_person) / 
                     (len(str(x).split()) + 1)
        )
        
        # Third-person pronoun ratio
        third_person = ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
                       'they', 'them', 'their', 'theirs', 'themselves', 'it', 'its', 'itself']
        features['third_person_ratio'] = X.apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in third_person) / 
                     (len(str(x).split()) + 1)
        )
        
        # Numbers ratio (specific numbers may indicate more factual statements)
        features['numbers_ratio'] = X.apply(
            lambda x: sum(c.isdigit() for c in str(x)) / (len(str(x)) + 1)
        )
        
        # Add spaCy features if available
        if HAS_SPACY:
            try:
                # Process texts in batches for efficiency
                docs = list(nlp.pipe([str(x) for x in X], disable=["parser"]))
                
                # Named entity ratio (more named entities may indicate more factual content)
                features['named_entity_ratio'] = [
                    len([ent for ent in doc.ents]) / (len(doc) + 1) for doc in docs
                ]
                
                # POS tag ratios
                features['noun_ratio'] = [
                    sum(1 for token in doc if token.pos_ == "NOUN") / (len(doc) + 1) for doc in docs
                ]
                
                features['verb_ratio'] = [
                    sum(1 for token in doc if token.pos_ == "VERB") / (len(doc) + 1) for doc in docs
                ]
                
                features['adj_ratio'] = [
                    sum(1 for token in doc if token.pos_ == "ADJ") / (len(doc) + 1) for doc in docs
                ]
                
                # Complexity measures
                features['sentence_count'] = [len(list(doc.sents)) for doc in docs]
                
            except Exception as e:
                print(f"Warning: Error extracting spaCy features: {e}")
            
        # Cast to float32 for better performance, especially on GPU
        features = features.astype(np.float32)
        
        return features.values

class ClaimVerifier:
    def __init__(self, use_gpu=HAS_CUDA, batch_size=128):
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.classifier = None
        self.pipeline = None
        self.best_model = None
        self.labels = None
        self.model_path = "claim_verifier_model.joblib"
        
        # Download required NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            print("Warning: Couldn't download NLTK resources. Some features may not work properly.")
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove punctuation but preserve special characters like question marks and exclamation points
        text = re.sub(r'[%s]' % re.escape(string.punctuation.replace('?', '').replace('!', '')), ' ', text)
        
        # Replace question marks and exclamation points with special tokens
        text = text.replace('?', ' QUESTIONMARK ')
        text = text.replace('!', ' EXCLAMATIONMARK ')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        
        # Remove some stop words but keep negation words as they change the meaning
        negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor',
                         'barely', 'hardly', 'rarely', 'scarcely', 'seldom', 'without', 'won\'t', 'wouldn\'t',
                         'can\'t', 'cannot', 'couldn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t'}
        filtered_stop_words = stop_words - negation_words
        
        # Use simple split instead of word_tokenize to avoid the punkt_tab error
        word_tokens = text.split()
        filtered_text = [word for word in word_tokens if word not in filtered_stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        stemmed_text = [stemmer.stem(word) for word in filtered_text]
        
        return ' '.join(stemmed_text)
    
    def extract_additional_features(self, df):
        """Extract additional features beyond standard text processing"""
        print("Extracting advanced features...")
        
        # Text-based features
        df['word_count'] = df['statement'].apply(lambda x: len(str(x).split()))
        df['char_count'] = df['statement'].apply(len)
        df['avg_word_length'] = df['statement'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Question mark count - questions might be less factual
        df['question_count'] = df['statement'].apply(lambda x: str(x).count('?'))
        
        # Exclamation mark count - emotional claims might be less factual
        df['exclamation_count'] = df['statement'].apply(lambda x: str(x).count('!'))
        
        # Number count - claims with specific numbers may be more factual
        df['number_count'] = df['statement'].apply(lambda x: sum(c.isdigit() for c in str(x)))
        
        # Capital letters percentage - ALL CAPS might indicate exaggeration
        df['capital_ratio'] = df['statement'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
        )
        
        # Quotation marks count - quoted content may indicate more factual information
        df['quote_count'] = df['statement'].apply(
            lambda x: str(x).count('"') + str(x).count("'")
        )
        
        # Named entity ratios - statements with more named entities may be more factual
        df['capitalized_word_ratio'] = df['statement'].apply(
            lambda x: sum(1 for word in str(x).split() if word and word[0].isupper()) / 
                     (len(str(x).split()) + 1)
        )
        
        # Sentiment features (using simple lexicon approach)
        positive_words = {'good', 'great', 'excellent', 'positive', 'nice', 'correct', 'true', 'right',
                         'best', 'better', 'proper', 'perfect', 'real', 'actual', 'factual', 'legitimate'}
        
        negative_words = {'bad', 'wrong', 'false', 'fake', 'incorrect', 'lie', 'lies', 'lying', 'hoax',
                         'conspiracy', 'misleading', 'misled', 'error', 'erroneous', 'deceptive'}
        
        df['positive_word_ratio'] = df['statement'].apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in positive_words) / 
                     (len(str(x).split()) + 1)
        )
        
        df['negative_word_ratio'] = df['statement'].apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in negative_words) / 
                     (len(str(x).split()) + 1)
        )
        
        return df
    
    def load_liar_dataset(self, train_path, test_path, valid_path=None):
        """Load and preprocess the LIAR dataset"""
        print("Loading and preprocessing the LIAR dataset...")
        
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
        
        # Preprocess statements with progress bar
        print("Preprocessing statements...")
        
        # Process in batches for better performance
        train_df['processed_statement'] = [self.preprocess_text(text) for text in tqdm(train_df['statement'], desc="Processing training data")]
        test_df['processed_statement'] = [self.preprocess_text(text) for text in tqdm(test_df['statement'], desc="Processing test data")]
        
        if valid_df is not None:
            valid_df['processed_statement'] = [self.preprocess_text(text) for text in tqdm(valid_df['statement'], desc="Processing validation data")]
        
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
                 
        # One-hot encode categorical features
        if valid_df is not None:
            dfs = [train_df, test_df, valid_df]
        else:
            dfs = [train_df, test_df]
                
        # Party affiliation
        for df in dfs:
            df['party_republican'] = df['party'].apply(lambda x: 1 if 'republican' in str(x).lower() else 0)
            df['party_democrat'] = df['party'].apply(lambda x: 1 if 'democrat' in str(x).lower() else 0)
            df['party_independent'] = df['party'].apply(lambda x: 1 if 'independent' in str(x).lower() else 0)
            
            # Subject categorization
            df['subject_politics'] = df['subject'].apply(lambda x: 1 if 'politic' in str(x).lower() else 0)
            df['subject_economy'] = df['subject'].apply(
                lambda x: 1 if any(term in str(x).lower() for term in ['economy', 'budget', 'tax', 'economic']) else 0
            )
            df['subject_health'] = df['subject'].apply(
                lambda x: 1 if any(term in str(x).lower() for term in ['health', 'medical', 'medicine', 'insurance']) else 0
            )
            df['subject_immigration'] = df['subject'].apply(
                lambda x: 1 if 'immigra' in str(x).lower() else 0
            )
        
        # Always use validation data for training to improve model performance
        if valid_df is not None:
            print("Using validation data for training to improve model performance...")
            combined_train_df = pd.concat([train_df, valid_df], ignore_index=True)
            return combined_train_df, test_df, None
        
        return train_df, test_df, valid_df
    
    def binarize_label(self, label):
        if label in ['true', 'mostly-true']:
            return 1  # True
        else:  # 'false', 'pants-fire', 'barely-true', 'half-true'
            return 0  # False
    
    def create_pipeline(self):
        """Create the model pipeline with GPU acceleration if available"""
        print("Creating optimized model pipeline...")
        
        # Text vectorization with enhanced parameters
        tfidf_vectorizer = TfidfVectorizer(
            max_features=20000,  # Increased for RTX 4090
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 3),
            sublinear_tf=True,
            use_idf=True,
            norm='l2'
        )
        
        # Create a count vectorizer for capturing specific patterns
        count_vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            max_features=15000,  # Increased for RTX 4090
            min_df=2,
            binary=True,
            token_pattern=r'\b\w+\b'
        )
        
        # Add our custom feature extractor
        text_features = FeatureUnion([
            ('tfidf', tfidf_vectorizer),
            ('count', count_vectorizer),
            ('advanced', AdvancedTextFeatureExtractor()),
        ])
        
        # Create the pipeline with GPU support if available
        if self.use_gpu:
            try:
                print("DIRECT GPU APPROACH: Creating CPU preprocessing with GPU model")
                
                # Create preprocessing pipeline (CPU-based)
                self.preprocess_pipeline = Pipeline([
                    ('features', text_features),
                    ('feature_selection', SelectKBest(f_classif, k=5000)),
                    ('svd', TruncatedSVD(n_components=400)),
                    ('scaler', StandardScaler())
                ])
                
                # Create GPU-based model separate from pipeline
                print("Initializing cuSVC for direct GPU training...")
                self.gpu_model = cuSVC(probability=True, kernel='rbf', C=10.0, gamma='scale', max_iter=5000)
                
                # Set a flag to indicate we're using direct GPU approach
                self.using_direct_gpu = True
                print("Using direct GPU approach with separate preprocessing pipeline and cuML model")
                
                # We'll implement a dummy pipeline for GridSearchCV compatibility
                self.pipeline = Pipeline([
                    ('features', text_features),
                    ('feature_selection', SelectKBest(f_classif, k=5000)),
                    ('svd', TruncatedSVD(n_components=400)),
                    ('scaler', StandardScaler()),
                    ('svm', SVC(probability=True))  # This won't actually be used
                ])
                
            except Exception as e:
                print(f"Error initializing direct GPU approach: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to CPU pipeline")
                self.use_gpu = False
                self.using_direct_gpu = False
                self._create_cpu_pipeline()
        else:
            self.using_direct_gpu = False
            self._create_cpu_pipeline()
    
    def _create_cpu_pipeline(self):
        """Create CPU-based pipeline with optimized parameters"""
        # Feature processors
        text_features = self.pipeline.named_steps['features'] if self.pipeline else None
        
        if text_features is None:
            # Text vectorization with enhanced parameters
            tfidf_vectorizer = TfidfVectorizer(
                max_features=15000,
                min_df=2,
                max_df=0.9,
                ngram_range=(1, 3),
                sublinear_tf=True,
                use_idf=True,
                norm='l2'
            )
            
            # Create a count vectorizer for capturing specific patterns
            count_vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                min_df=2,
                binary=True,
                token_pattern=r'\b\w+\b'
            )
            
            # Add our custom feature extractor
            text_features = FeatureUnion([
                ('tfidf', tfidf_vectorizer),
                ('count', count_vectorizer),
                ('advanced', AdvancedTextFeatureExtractor()),
            ])
        
        # Create multi-model ensemble for better performance
        svm_rbf = SVC(probability=True, kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', max_iter=2000)
        svm_linear = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000)
        
        # Create the final pipeline
        self.pipeline = Pipeline([
            ('features', text_features),
            ('feature_selection', SelectKBest(f_classif, k=3000)),
            ('svd', TruncatedSVD(n_components=300)),
            ('scaler', StandardScaler()),
            ('svm', svm_rbf)
        ])
    
    def train(self, train_df):
        """Train the model with comprehensive hyperparameter optimization"""
        start_time = time.time()
        
        # Create the pipeline if it doesn't exist
        if self.pipeline is None:
            self.create_pipeline()
        
        # Get the features and target
        X_text = train_df['processed_statement']
        y = train_df['binary_label']
        
        if hasattr(self, 'using_direct_gpu') and self.using_direct_gpu:
            print("\n*** USING DIRECT GPU TRAINING ***")
            
            # First, preprocess the data using sklearn pipeline
            print("Preprocessing data with CPU pipeline...")
            X_processed = self.preprocess_pipeline.fit_transform(X_text, y)
            print(f"Processed data shape: {X_processed.shape}")
            
            # Force data to 32-bit float for GPU
            X_processed = X_processed.astype(np.float32)
            
            # Now directly train on GPU
            print("Training SVM directly on GPU...")
            gpu_start = time.time()
            self.gpu_model.fit(X_processed, y)
            gpu_time = time.time() - gpu_start
            print(f"GPU SVM training completed in {gpu_time:.2f} seconds")
            
            # Store the best model
            self.best_model = self.gpu_model
            self.classifier = self.gpu_model
            
            # Save the model
            self.save_model(self.model_path)
            
            # Report training time
            training_time = (time.time() - start_time) / 60
            print(f"\nTotal training completed in {training_time:.2f} minutes")
            
            return self.gpu_model
        
        # Create stratified k-fold for more robust evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Comprehensive parameter grid optimized for RTX 4090
        if self.use_gpu:
            # Optimized grid for RTX 4090 GPU
            param_grid = {
                'svm__C': [1, 5, 10, 20, 50],
                'svm__gamma': ['scale', 0.1, 0.01, 0.001],
                'feature_selection__k': [3000, 5000, 7000],
                'svd__n_components': [300, 400, 500]
            }
        else:
            # Standard grid for CPU
            param_grid = {
                'svm__C': [0.1, 1, 5, 10, 20, 50],
                'svm__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                'svm__kernel': ['rbf', 'poly', 'sigmoid'],
                'feature_selection__k': [2000, 3000, 4000],
                'feature_selection__score_func': [f_classif, mutual_info_classif],
                'svd__n_components': [200, 300, 400]
            }
        
        print("\nPerforming hyperparameter optimization...")
        print(f"Using {cv.n_splits} fold cross-validation")
        print(f"Parameter grid: {param_grid}")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            verbose=2,
            n_jobs=1 if self.use_gpu else -1  # Always single job for GPU to avoid memory issues
        )
        
        print("\nTraining the model (this may take a while)...")
        grid_search.fit(X_text, y)
        
        # Get best model and parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\nBest parameters found: {best_params}")
        print(f"Best cross-validation accuracy: {best_score:.4f}")
        
        # Store the best model
        self.pipeline = grid_search.best_estimator_
        self.best_model = grid_search.best_estimator_
        
        # Train final model on all data
        print("\nTraining final model on all data with best parameters...")
        self.pipeline.fit(X_text, y)
        
        # Get the classifier from the pipeline
        self.classifier = self.pipeline.named_steps['svm']
        
        # Report training time
        training_time = (time.time() - start_time) / 60
        print(f"\nTraining completed in {training_time:.2f} minutes")
        
        # Save the model
        self.save_model(self.model_path)
        
        return self.pipeline
    
    def evaluate(self, test_df):
        """Evaluate the model on test data"""
        if self.pipeline is None and not hasattr(self, 'gpu_model'):
            if os.path.exists(self.model_path):
                self.load_model(self.model_path)
            else:
                raise ValueError("Model has not been trained yet and no saved model found")
        
        # Get text features
        X_test_text = test_df['processed_statement']
        
        # Target
        y_test = test_df['binary_label']
        
        # Time the prediction
        start_time = time.time()
        
        # Make predictions - check if we're using direct GPU approach
        if hasattr(self, 'using_direct_gpu') and self.using_direct_gpu:
            # First preprocess the data
            X_processed = self.preprocess_pipeline.transform(X_test_text)
            X_processed = X_processed.astype(np.float32)
            
            # Make predictions with GPU model
            predictions = self.gpu_model.predict(X_processed)
        else:
            # Use regular pipeline
            predictions = self.pipeline.predict(X_test_text)
        
        prediction_time = time.time() - start_time
        print(f"Prediction time for {len(X_test_text)} samples: {prediction_time:.4f} seconds")
        print(f"Average prediction time per sample: {prediction_time/len(X_test_text)*1000:.4f} ms")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
        
        # Print evaluation metrics
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        # Print summary metrics
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return predictions, accuracy
    
    def predict_claim(self, claim):
        """Predict whether a claim is true or false"""
        if self.pipeline is None and not hasattr(self, 'gpu_model'):
            if os.path.exists(self.model_path):
                self.load_model(self.model_path)
            else:
                raise ValueError("Model has not been trained or loaded yet")
            
        # Preprocess the claim
        processed_claim = self.preprocess_text(claim)
        
        # Transform to features
        X_text = pd.Series([processed_claim])
        
        # Check if we're using direct GPU approach
        if hasattr(self, 'using_direct_gpu') and self.using_direct_gpu:
            # Use preprocessing pipeline and GPU model separately
            X_processed = self.preprocess_pipeline.transform(X_text)
            X_processed = X_processed.astype(np.float32)
            
            # Make prediction with GPU model
            prediction = self.gpu_model.predict(X_processed)[0]
            prob = self.gpu_model.predict_proba(X_processed)[0]
        else:
            # Use regular pipeline
            prediction = self.pipeline.predict(X_text)[0]
            prob = self.pipeline.predict_proba(X_text)[0]
        
        # Convert numerical prediction back to label
        if prediction == 1:
            label = "true"
            confidence = prob[1]
        else:
            label = "false"
            confidence = prob[0]
        
        return label, confidence
    
    def save_model(self, model_path):
        """Save the model to a file"""
        if not hasattr(self, 'using_direct_gpu') or not self.using_direct_gpu:
            if self.pipeline is None:
                raise ValueError("No model to save")
            joblib.dump(self.pipeline, model_path)
        else:
            # Save both preprocessing pipeline and GPU model
            model_data = {
                'preprocess_pipeline': self.preprocess_pipeline,
                'gpu_model': self.gpu_model,
                'using_direct_gpu': True
            }
            joblib.dump(model_data, model_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load the model from a file"""
        try:
            # Try to load as direct GPU format first
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict) and 'using_direct_gpu' in model_data:
                print("Loading direct GPU model format")
                self.preprocess_pipeline = model_data['preprocess_pipeline']
                self.gpu_model = model_data['gpu_model']
                self.using_direct_gpu = True
                self.classifier = self.gpu_model
                print(f"Direct GPU model loaded from {model_path}")
                return
        except:
            pass
        
        # Regular pipeline format
        self.pipeline = joblib.load(model_path)
        self.using_direct_gpu = False
        print(f"Pipeline model loaded from {model_path}")
        
        # Update classifier reference for easier access
        if 'svm' in self.pipeline.named_steps:
            self.classifier = self.pipeline.named_steps['svm']
        elif 'linear_svc' in self.pipeline.named_steps:
            self.classifier = self.pipeline.named_steps['linear_svc']

if __name__ == "__main__":
    # Initialize the claim verifier with GPU support if available
    verifier = ClaimVerifier(use_gpu=HAS_CUDA)
    
    # Print GPU information
    if HAS_CUDA or PYTORCH_CUDA:
        if PYTORCH_CUDA:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        
        if HAS_CUDA:
            # Check RAPIDS version
            try:
                print(f"RAPIDS cuML version: {cuml.__version__}")
                # Test if basic cuML operations work
                import numpy as np
                test_data = np.random.random((10, 5)).astype(np.float32)
                test_labels = np.random.randint(0, 2, 10)
                test_model = cuSVC()
                print("Testing basic cuSVC functionality...")
                test_model.fit(test_data, test_labels)
                print("Basic cuML test passed!")
            except Exception as e:
                print(f"Error testing cuML: {e}")
                import traceback
                traceback.print_exc()
                print("WARNING: Will fall back to CPU")
                HAS_CUDA = False
                verifier.use_gpu = False
    
    # Load and preprocess the LIAR dataset
    train_df, test_df, valid_df = verifier.load_liar_dataset(
        'Liar/train.tsv',
        'Liar/test.tsv',
        'Liar/valid.tsv'
    )
    
    # Train the model synchronously (no async mode)
    print("\nTraining the model (this may take a while)...")
    verifier.train(train_df)
    
    # Evaluate the model on test set
    print("\nEvaluating the model on test set...")
    test_predictions, test_accuracy = verifier.evaluate(test_df)
    
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