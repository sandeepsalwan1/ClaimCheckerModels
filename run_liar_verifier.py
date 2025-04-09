import sys
import pandas as pd
import numpy as np
import time
from claim_verifier import ClaimVerifier

def main():
    print("Starting LIAR dataset claim verification with Optimized Kernel SVM...")
    
    # Initialize the claim verifier
    verifier = ClaimVerifier()
    
    # Load and preprocess the LIAR dataset
    print("Loading and preprocessing the LIAR dataset...")
    start_time = time.time()
    train_df, test_df, valid_df = verifier.load_liar_dataset(
        'Liar/train.tsv',
        'Liar/test.tsv',
        'Liar/valid.tsv'
    )
    print(f"Dataset loading and preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    print(f"Dataset loaded: {len(train_df)} training examples, {len(test_df)} test examples, {len(valid_df)} validation examples")
    
    # Print label distribution
    print("\nLabel distribution in training set:")
    print(train_df['label'].value_counts())
    print("\nBinary label distribution in training set:")
    print(train_df['binary_label'].value_counts())
    
    # Print feature statistics
    print("\nAdditional feature statistics:")
    feature_columns = ['word_count', 'char_count', 'question_count', 
                       'exclamation_count', 'number_count', 'capital_ratio',
                       'credibility_score']
    
    print(train_df[feature_columns].describe())
    
    # Train the model with advanced features and kernel optimization
    print("\nTraining the SVM model with optimized kernels...")
    train_start = time.time()
    verifier.train(train_df)
    train_time = time.time() - train_start
    print(f"Model training completed in {train_time:.2f} seconds")
    
    # Evaluate the model on test set
    print("\nEvaluating the model on test set...")
    test_predictions, test_accuracy = verifier.evaluate(test_df)
    
    # Evaluate the model on validation set
    print("\nEvaluating the model on validation set...")
    valid_predictions, valid_accuracy = verifier.evaluate(valid_df)
    
    # Save the trained model
    model_path = 'claim_verifier_model.joblib'
    model_save_start = time.time()
    verifier.save_model(model_path)
    print(f"\nModel saved to {model_path} in {time.time() - model_save_start:.2f} seconds")
    
    # Test on some examples from the dataset
    print("\nTesting on examples from the dataset:")
    
    # Get some examples from test set with different labels
    true_examples = test_df[test_df['label'] == 'true'].sample(min(2, len(test_df[test_df['label'] == 'true'])))
    false_examples = test_df[test_df['label'] == 'false'].sample(min(2, len(test_df[test_df['label'] == 'false'])))
    pants_fire_examples = test_df[test_df['label'] == 'pants-fire'].sample(min(2, len(test_df[test_df['label'] == 'pants-fire'])))
    
    sample_df = pd.concat([true_examples, false_examples, pants_fire_examples])
    
    for _, row in sample_df.iterrows():
        claim = row['statement']
        true_label = row['label']
        
        predict_start = time.time()
        prediction, confidence = verifier.predict_claim(claim)
        predict_time = time.time() - predict_start
        
        print(f"\nClaim: {claim}")
        print(f"True label: {true_label}")
        print(f"Predicted: {prediction} (confidence: {confidence:.4f})")
        print(f"Prediction time: {predict_time*1000:.2f} ms")
    
    # Test on custom claims
    print("\nTesting custom claims:")
    custom_claims = [
        "Building a wall on the U.S.-Mexico border will take literally years.",
        "The number of illegal immigrants could be 3 million. It could be 30 million.",
        "Democrats already agreed to a deal that Republicans wanted.",
        "Austin has more lobbyists working for it than any other municipality in Texas."
    ]
    
    all_predict_times = []
    for claim in custom_claims:
        predict_start = time.time()
        verdict, confidence = verifier.predict_claim(claim)
        predict_time = time.time() - predict_start
        all_predict_times.append(predict_time)
        
        print(f"\nClaim: {claim}")
        print(f"Verdict: {verdict} (confidence: {confidence:.4f})")
        print(f"Prediction time: {predict_time*1000:.2f} ms")
    
    # Compare model performance
    print("\nModel Performance Summary:")
    print(f"Test set accuracy: {test_accuracy:.4f}")
    print(f"Validation set accuracy: {valid_accuracy:.4f}")
    print(f"Average prediction time: {(sum(all_predict_times)/len(all_predict_times))*1000:.2f} ms")
    print(f"Total training time: {train_time:.2f} seconds")
    
    print("\nThis optimized kernel SVM model uses:")
    print("- TF-IDF features with bigrams")
    print("- Feature selection and dimensionality reduction")
    print("- Optimized SVM with advanced kernel selection")
    print("- Kernel approximation techniques for faster prediction (if beneficial)")
    print("- Additional linguistic and metadata features")
    print("- Stratified K-fold cross-validation")
    print("- Grid search for extensive hyperparameter tuning")

if __name__ == "__main__":
    main() 