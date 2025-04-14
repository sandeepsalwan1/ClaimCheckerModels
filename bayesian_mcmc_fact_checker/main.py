import numpy as np
import requests
import json
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


# NEWS_API_KEY = "5999f11813dc487ea533351217e73ffb"

# news_api_url = (f'https://newsapi.org/v2/top-headlines?q=politics&from=2025-03-09&sortBy=popularity&apiKey={NEWS_API_KEY}')

# news_api_data = requests.get(news_api_url).json()
# print(news_api_data['articles'][0]['content'])  

# factcheck_url = 'https://www.factcheck.org/latest-articles/'
# factcheck_response = requests.get(factcheck_url)
# soup = BeautifulSoup(factcheck_response.text, 'html.parser')
# factcheck_articles = soup.find_all('article', class_='post')

# try:
#     p = Politifact()
#     response = p.statements().people('Barack Obama')

#     if hasattr(response, 'text') and response.text.strip():
#         obama_related = response.json()
#         for statement in obama_related:
#             print(statement.ruling)
#     else:
#         print("Empty response from Politifact API")

# except json.JSONDecodeError as e:
#     print(f"Error parsing Politifact API response: {e}")

# except Exception as e:
#     print(f"Error with Politifact API request: {e}")



"""LIAR Dataset"""

# don't need this for now if vectorizing, but keep if necesary
def preprocess_text(text):
    # make it so that the numbers don't get taken out
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in text]
    return ''.join(tokens)

# print(preprocess_text("On October 1, 2023, the president announced a new policy."))  # Example usage

def run_liar_dataset(test=False):
    st = SentenceTransformer('all-MiniLM-L6-v2')
    # emb = st.encode(["On October 1, 2023, the president announced a new policy."])
    # print(emb)
    # print(emb.shape)

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

    # perform PCA to reduce to 50 dimensions for now
    # later set it back to 100 or higher for more accuracy
    pca = PCA(n_components=50)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    bc_model = BayesianClassifier(num_classes=6, num_features=X_train.shape[1])

    # acceptance_rates = {}
    # accuracies = {}

    if (test):
        print("Using test dataset")

    hmc = HamiltonianMonteCarlo(model=bc_model, num_samples=1000, step_size=0.01, num_steps=10, burn_in=200)

    # run HMC sampling
    weight, alpha, log_post, acceptance_rate = hmc.hmc_sampling(X_train, y_train)


    print(f"Weights: {weight}")
    print(f"Alpha: {alpha}")
    print(f"Log Posterior: {log_post}")
    print(f"Successfully collected {len(weight)} samples")
    print(f'Acceptance rate: {acceptance_rate}')
    # acceptance_rates[i] = acceptance_rate

    # used gen-AI to debug - fixed weight udpates
    mean_weights = np.mean(weight, axis=0)
    mean_alpha = np.mean(alpha, axis=0)

    bc_model.weights = mean_weights
    bc_model.alpha = mean_alpha

    X_test_set = X_test if test else X_val
    y_test_set = y_test if test else y_val

    y_pred = bc_model.predict(X_test_set)

    print("Validation set predictions:")
    print(y_pred)
    print(classification_report(y_test_set, y_pred))

    # def calculate_accuracies(y_true, y_pred):
    #     exact_predictions = np.sum(y_true == y_pred)
    #     rough_predictions = np.sum(np.abs(y_true - y_pred) <= 1)
    #     total_predictions = len(y_true)
    #     exact_accuracy = exact_predictions / total_predictions
    #     rough_accuracy = rough_predictions / total_predictions
    #     return exact_accuracy, rough_accuracy

    # exact_accuracy, rough_accuracy = calculate_accuracies(y_val, y_pred)
    # print(f"Exact accuracy: {exact_accuracy}")
    # print(f"Accuracy within +/- 1 category: {rough_accuracy}")
    # accuracies[i] = [exact_accuracy, rough_accuracy]

    # calculate accuracies
    y_test_set_binary = np.where(y_test_set > 2, 1, 0)
    y_pred_binary = np.where(y_pred > 2, 1, 0)
    accuracy = np.sum(y_test_set_binary == y_pred_binary) / len(y_test_set_binary)
    # accuracies[i] = exact_accuracy

    # confusion matrix
    cm = confusion_matrix(y_test_set_binary, y_pred_binary)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['True', 'False'], rotation=45)
    plt.yticks(tick_marks, ['True', 'False'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_bin2_test.png')

    y_probs = bc_model.predict_probs(X_test_set)[:, 1]
    y_true = y_test_set_binary

    thresholds = np.arange(0, 1.01, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    # TODO: find the optimal threshold

    # f1 score vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='blue')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold')
    # plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.legend()
    plt.grid()
    plt.savefig('f1_vs_threshold.png')
    plt.show()

    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.savefig('roc_curve.png')
    plt.show()

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
