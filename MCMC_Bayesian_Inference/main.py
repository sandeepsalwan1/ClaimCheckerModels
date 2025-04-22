import numpy as np
import requests
from politifact import Politifact
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import regex as re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from bayesian_classifier import BayesianClassifier
from hmc import HamiltonianMonteCarlo
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from baseline_comparison import BaselineLogisticRegression
from baseline_comparison import compare_with_mcmc, load_liar_dataset

# preprocessing
def preprocess_text(text):
    # make it so that the numbers don't get taken out
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in text]
    return ''.join(tokens)

def run_liar_dataset(test=False):
    st = SentenceTransformer('all-MiniLM-L6-v2')
    # loading LIAR dataset

    # train
    liar_data_train = pd.read_csv('liar/train.tsv', sep='\t', header=None)
    liar_data_train.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
    statements_train = liar_data_train[['label', 'statement']]
    # counts_train = liar_data_train[['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']]
    print("Loaded training data")

    # test
    liar_data_test = pd.read_csv('liar/test.tsv', sep='\t', header=None)
    liar_data_test.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
    statements_test = liar_data_test[['label', 'statement']]
    # counts_test = liar_data_test[['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']]
    print("Loaded test data")

    # validation
    liar_data_val = pd.read_csv('liar/valid.tsv', sep='\t', header=None)
    liar_data_val.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
    statements_val = liar_data_val[['label', 'statement']]
    # counts_val = liar_data_val[['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']]
    print("Loaded validation data")

    statements_train['statement'] = statements_train['statement'].apply(preprocess_text)
    statements_test['statement'] = statements_test['statement'].apply(preprocess_text)
    statements_val['statement'] = statements_val['statement'].apply(preprocess_text)
    print("Preprocessed statements")

    label_dict = {
        'pants-fire': 0,
        'false': 1,
        'barely-true': 2,
        'half-true': 3,
        'mostly-true': 4,
        'true': 5
    }

    # first time, save embeddings to file
    # after that, just load them from file

    # if file doesnt exist
    if not (os.path.exists('liar/train_embeddings.npy') and os.path.exists('liar/test_embeddings.npy') and os.path.exists('liar/valid_embeddings.npy')):
        print("Embeddings not found, creating new ones")

        # use sentence transformer to get embeddings
        train_embeddings = np.array([st.encode(text) for text in statements_train['statement']])
        test_embeddings = np.array([st.encode(text) for text in statements_test['statement']])
        val_embeddings = np.array([st.encode(text) for text in statements_val['statement']])
        print("Encoded training data")

        # numeric labels
        train_labels = np.array(statements_train['label'].map(label_dict))
        test_labels = np.array(statements_test['label'].map(label_dict))
        val_labels = np.array(statements_val['label'].map(label_dict))
        print("Encoded labels")

        # save embeddings and labels as numpy arrays
        np.save('liar/train_embeddings.npy', train_embeddings)
        np.save('liar/test_embeddings.npy', test_embeddings)
        np.save('liar/val_embeddings.npy', val_embeddings)
        np.save('liar/train_labels.npy', train_labels)
        np.save('liar/test_labels.npy', test_labels)
        np.save('liar/val_labels.npy', val_labels)
        print("Saved embeddings to file")
        
        # Update your dataframe for continued use in this session
        statements_train['statement'] = list(train_embeddings)
        statements_test['statement'] = list(test_embeddings)
        statements_val['statement'] = list(val_embeddings)
        statements_train['label'] = train_labels
        statements_test['label'] = test_labels
        statements_val['label'] = val_labels
    else:
        print("Embeddings found, loading from file")

        # load embeddings from numpy files
        train_embeddings = np.load('liar/train_embeddings.npy')
        test_embeddings = np.load('liar/test_embeddings.npy')
        val_embeddings = np.load('liar/val_embeddings.npy')
        train_labels = np.load('liar/train_labels.npy')
        test_labels = np.load('liar/test_labels.npy')
        val_labels = np.load('liar/val_labels.npy')
        print("Loaded embeddings from file")
        
        # Update your dataframe for continued use in this session
        statements_train['statement'] = list(train_embeddings)
        statements_test['statement'] = list(test_embeddings)
        statements_val['statement'] = list(val_embeddings)
        statements_train['label'] = train_labels
        statements_test['label'] = test_labels
        statements_val['label'] = val_labels

    X_train = np.array(statements_train['statement'].tolist())
    y_train = np.array(statements_train['label'].tolist())
    X_val = np.array(statements_val['statement'].tolist())
    y_val = np.array(statements_val['label'].tolist())
    X_test = np.array(statements_test['statement'].tolist())
    y_test = np.array(statements_test['label'].tolist())

    # currently not being used, but used for debugging purposes
    # use if it is necessary to increase the training speed
    pca = PCA(n_components=150)
    # X_train = pca.fit_transform(X_train)
    # X_val = pca.transform(X_val)
    # X_test = pca.transform(X_test)

    bc_model = BayesianClassifier(num_classes=6, num_features=X_train.shape[1])

    if (test):
        print("Using test dataset")

    hmc = HamiltonianMonteCarlo(model=bc_model, num_samples=5000, step_size=0.01, num_steps=10, burn_in=1000)

    # run HMC sampling
    weight, alpha, log_post, acceptance_rate = hmc.hmc_sampling(X_train, y_train)


    print(f"Weights: {weight}")
    print(f"Alpha: {alpha}")
    print(f"Log Posterior: {log_post}")
    print(f"Successfully collected {len(weight)} samples")
    print(f'Acceptance rate: {acceptance_rate}')

    mean_weights = np.mean(weight, axis=0)
    mean_alpha = np.mean(alpha, axis=0)

    bc_model.weights = mean_weights
    bc_model.alpha = mean_alpha

    X_test_set = X_test if test else X_val
    y_test_set = y_test if test else y_val

    y_pred = bc_model.predict(X_test_set)

    # calculate accuracies
    y_test_set_binary = np.where(y_test_set > 2, 1, 0)
    y_pred_binary = np.where(y_pred > 2, 1, 0)
    accuracy = np.sum(y_test_set_binary == y_pred_binary) / len(y_test_set_binary)

    # confusion matrix
    cm = confusion_matrix(y_test_set_binary, y_pred_binary)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['False', 'True'], rotation=45)
    plt.yticks(tick_marks, ['False', 'True'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_bin3_{"test" if test else "val"}.png')

    y_probs_all = bc_model.predict_probs(X_test_set)
    y_true_binary = np.where(y_test_set > 2, 1, 0)
    y_pred_binary_probs = y_probs_all[:, 3:].sum(axis=1)

    print(f"{"Test" if test else "Validation"} set predictions:")
    print(y_pred)

    print(classification_report(y_true_binary, y_pred_binary, target_names=['False', 'True']))

    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary_probs)
    auc_score = roc_auc_score(y_true_binary, y_pred_binary_probs)

    thresholds = np.arange(0, 1.01, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_binary_probs >= threshold).astype(int)
        f1 = f1_score(y_true_binary, y_pred)
        f1_scores.append(f1)

    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Optimal threshold: {optimal_threshold}")

    # f1 score vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='blue')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.legend()
    plt.grid()
    plt.savefig(f'f1_vs_threshold_{"test" if test else "val"}.png')

    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.savefig(f'roc_curve_{"test" if test else "val"}.png')

    # new confusion matrix with optimal threshold
    y_pred_optimal = (y_pred_binary_probs >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_true_binary, y_pred_optimal)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm_optimal, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix with Optimal Threshold')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['False', 'True'], rotation=45)
    plt.yticks(tick_marks, ['False', 'True'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_optimal_{"test" if test else "val"}.png')

    # another confusion matrix with default threshold
    y_pred_optimal = (y_pred_binary_probs >= 0.5).astype(int)
    cm_optimal = confusion_matrix(y_true_binary, y_pred_optimal)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm_optimal, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix with Optimal Threshold')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['False', 'True'], rotation=45)
    plt.yticks(tick_marks, ['False', 'True'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_def_{"test" if test else "val"}.png')

    print(f"{"Test" if test else "Validation"} set predictions (optimal):")
    print(y_pred)

    print(classification_report(y_true_binary, y_pred_optimal, target_names=['False', 'True']))

    # confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['False', 'True']).plot(ax=axes[0], cmap='Blues')
    ConfusionMatrixDisplay(confusion_matrix=cm_optimal, display_labels=['False', 'True']).plot(ax=axes[1], cmap='Blues')
    axes[0].set_title('Confusion Matrix (Default Threshold = 0.5)')
    axes[1].set_title(f'Confusion Matrix (Optimal Threshold = {optimal_threshold:.2f})')
    plt.savefig(f'confusion_matrices_combined_{"test" if test else "val"}.png')

    print(f'AUC Score: {auc_score}')

    print(f'Acceptance rate: {acceptance_rate}')
    print(f'Accuracy: {float(accuracy)}')

    
    # Save the model after training
    model_filename = 'bayesian_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(bc_model, f)
    print(f"Model saved to {model_filename}")

    # Save the PCA model
    pca_filename = 'pca_model.pkl'
    with open(pca_filename, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA model saved to {pca_filename}")
    
    return mean_weights, mean_alpha

def load_model_and_predict(text):
    # Load the Bayesian model
    with open('bayesian_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    
    # preprocessing
    st = SentenceTransformer('all-MiniLM-L6-v2')
    preprocessed_text = preprocess_text(text)
    text_embedding = st.encode([preprocessed_text])

    # PCA
    with open('pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    text_embedding_reduced = pca.transform(text_embedding)

    # make prediction
    prediction = loaded_model.predict(text_embedding_reduced)
    prediction_binary = 1 if prediction > 2 else 0
    return prediction_binary

def break_article_into_sentences(article):
    # Split the article into sentences
    sentences = re.split(r'(?<=[.!?]) +', article)

    for i, sentence in enumerate(sentences):
        # Preprocess each sentence
        sentences[i] = preprocess_text(sentence)

    return sentences

def predict_article(article):
    sentences = break_article_into_sentences(article)
    scores = []
    for sentence in sentences:
        print(f"Processing sentence: {sentence}")
        prediction = load_model_and_predict(sentence)
        scores.append(prediction)

    credibility_score = np.mean(scores)
    print(f"Credibility score: {round(credibility_score, 2) * 100}%")

if __name__ == "__main__":
    weights, alpha = run_liar_dataset(test=True)
    # article = """"""
    # predict_article(article)
    
    # baseline comparisons
    baseline = BaselineLogisticRegression()
    train_df, test_df, valid_df = load_liar_dataset(
        'Liar/train.tsv',
        'Liar/test.tsv',
        'Liar/valid.tsv'
    )
    baseline.train(train_df)
    baseline_metrics = baseline.evaluate(test_df)
    baseline.save_model('baseline_model.joblib')
    fpr, tpr, _ = roc_curve(test_df['binary_label'], baseline_metrics['prediction_probs'])
    baseline_metrics['fpr'] = fpr
    baseline_metrics['tpr'] = tpr

    compare_with_mcmc(baseline_metrics)

