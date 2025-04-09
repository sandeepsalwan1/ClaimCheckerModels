# Advanced Claim Verifier - Optimized Kernel SVM for LIAR Dataset

This project implements an advanced Support Vector Machine (SVM) model with optimized kernels to classify claims as true or false based on the LIAR dataset.

## Overview

The LIAR dataset contains political statements labeled with varying degrees of truthfulness: pants-fire, false, barely-true, half-true, mostly-true, and true. This project simplifies the task to a binary classification problem by grouping "true" and "mostly-true" as "true", and the rest as "false".

## Advanced Features

- **Advanced text preprocessing**: Special handling of negation words and punctuation
- **Rich feature extraction**: Word count, character count, question marks, exclamation points, numeric content, capitalization
- **TF-IDF vectorization with bigrams**: Capturing more complex phrases
- **Feature selection**: Using chi-squared method to select the most relevant features
- **Dimensionality reduction**: Using TruncatedSVD (similar to LSA) to reduce feature space
- **Optimized kernel selection**: Testing linear, RBF, polynomial, and sigmoid kernels to capture non-linear patterns
- **Kernel approximation techniques**: Using RBFSampler and Nystroem methods for faster prediction
- **Historical credibility features**: Using speaker's past reliability
- **Extensive hyperparameter optimization**: Grid search with stratified k-fold cross-validation
- **Performance monitoring**: Tracking training and prediction times
- **Model evaluation**: Using accuracy, precision, recall, and F1-score
- **Confidence scores**: Providing confidence levels for predictions

## Kernel Optimization

This project incorporates advanced kernel optimization techniques:

1. **Extended Kernel Selection**: Tests multiple kernels (linear, RBF, polynomial, sigmoid) to find the optimal choice for claim verification.

2. **Kernel Approximation**: For RBF kernels, the model evaluates potential speedups using approximation techniques:
   - **RBFSampler**: Uses random Fourier features to approximate the RBF kernel
   - **Nystroem Method**: Uses sampling to approximate kernel feature maps

3. **Tuned Hyperparameters**:
   - **Degree**: Optimized for polynomial kernels (2-4)
   - **Coef0**: Tuned for polynomial and sigmoid kernels
   - **Gamma**: More granular search for RBF, polynomial, and sigmoid kernels
   - **C**: Regularization parameter optimized across various values

4. **Performance Monitoring**: Tracks training and prediction times to ensure optimal speed-accuracy tradeoff.

## Project Structure

- `claim_verifier.py`: Main class implementing the optimized kernel SVM model
- `run_liar_verifier.py`: Script to train, evaluate, and test the model
- `verify_claim.py`: Interactive tool for verifying user-entered claims
- `requirements.txt`: Package dependencies
- `/Liar`: Directory containing the LIAR dataset files (train.tsv, test.tsv, valid.tsv)

## Getting Started

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Train and evaluate the model:
   ```
   python run_liar_verifier.py
   ```
4. Verify your own claims:
   ```
   python verify_claim.py
   ```

## Advanced Model Details

The model uses a pipeline with several components:

1. **TF-IDF Vectorization**: 
   - Converts text to numerical features
   - Includes both unigrams and bigrams
   - Uses sublinear TF scaling
   - Applies min_df and max_df parameters to filter rare and common terms

2. **Feature Selection**:
   - Selects the most informative features using chi-squared test
   - Optimized number of features through grid search

3. **Dimensionality Reduction**:
   - Applies TruncatedSVD (similar to Latent Semantic Analysis)
   - Reduces feature space to manageable dimensions
   - Helps capture semantic relationships between words

4. **Feature Scaling**:
   - Standardizes features for better SVM performance

5. **SVM with Optimized Kernels**:
   - Tests multiple kernels (linear, RBF, polynomial, sigmoid)
   - Optimizes C parameter for regularization strength
   - Tunes gamma parameter for non-linear kernels
   - Optimizes degree for polynomial kernels
   - Tunes coef0 for polynomial and sigmoid kernels
   - Applies kernel approximation for faster prediction when beneficial
   - Uses probability calibration for confidence scores

## Performance Optimization

The model includes several performance optimizations:

1. **Kernel Approximation**: For large datasets, exact kernel computations can be slow. We automatically evaluate if kernel approximation techniques (RBFSampler, Nystroem) can provide faster prediction without significant accuracy loss.

2. **Stratified K-Fold Cross-Validation**: Ensures balanced class representation during validation, leading to more robust hyperparameter selection.

3. **Parallel Processing**: Uses all available CPU cores during grid search for faster training.

4. **Performance Monitoring**: Tracks and reports training and prediction times to ensure efficiency.

## Dataset

The LIAR dataset contains 12,800+ human-labeled short statements from POLITIFACT.COM's API, with their truthfulness labels and metadata. The dataset is split into training, validation, and test sets.

## Performance

The model's performance metrics (accuracy, precision, recall, and F1-score) are displayed when running the evaluation. The grid search cross-validation helps identify the best parameters for optimal performance. Performance timing information is also displayed to assess prediction speed.

## Example Usage

After training, you can verify claims using:

```python
from claim_verifier import ClaimVerifier
import time

verifier = ClaimVerifier()
verifier.load_model('claim_verifier_model.joblib')

claim = "Your claim text here"
start_time = time.time()
verdict, confidence = verifier.predict_claim(claim)
predict_time = time.time() - start_time
print(f"Verdict: {verdict} (confidence: {confidence:.4f})")
print(f"Prediction time: {predict_time*1000:.2f} ms")
```

## License

This project uses the LIAR dataset, which should be cited appropriately when used.

## References

William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, ACNLP 2017

