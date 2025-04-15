# Save as test_gpu.py
import cuml
import numpy as np
import time
from cuml.svm import SVC as cuSVC
from cuml.decomposition import TruncatedSVD as cuTruncatedSVD
from cuml.preprocessing import StandardScaler as cuStandardScaler
import torch
from sklearn.svm import SVC

print("\n=== GPU SVM ACCELERATION TEST ===\n")

# Check basic CUDA functionality
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"cuML version: {cuml.__version__}")

# Generate random data
print("Generating test data...")
n_samples = 20000
n_features = 200
X = np.random.random((n_samples, n_features)).astype(np.float32)
y = np.random.randint(0, 2, n_samples)

# First run GPU test
print("\n=== TESTING GPU PERFORMANCE ===")
gpu_start = time.time()
gpu_model = cuSVC(kernel='rbf', C=10.0)
print("Training SVM on GPU...")
gpu_model.fit(X, y)
gpu_time = time.time() - gpu_start
print(f"GPU training time: {gpu_time:.2f} seconds")

# Then run CPU test
print("\n=== TESTING CPU PERFORMANCE ===")
cpu_start = time.time()
cpu_model = SVC(kernel='rbf', C=10.0)
print("Training SVM on CPU...")
cpu_model.fit(X, y)
cpu_time = time.time() - cpu_start
print(f"CPU training time: {cpu_time:.2f} seconds")

# Calculate speedup
speedup = cpu_time / gpu_time
print(f"\n=== RESULTS ===")
print(f"CPU time: {cpu_time:.2f} seconds")
print(f"GPU time: {gpu_time:.2f} seconds")
print(f"GPU speedup: {speedup:.2f}x faster")

# Force GPU memory usage for nvidia-smi
print("\nForcing GPU memory allocation to verify in nvidia-smi...")
X_gpu = torch.tensor(X, device='cuda')
print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\nRun nvidia-smi in another terminal to check GPU usage")
print("Waiting 5 seconds before exiting...")
time.sleep(5)

# Check if GPU is being properly utilized
print("\nDiagnostic information:")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Force more GPU usage to verify
print("\nRunning additional GPU test...")
X_gpu = torch.tensor(X_gpu, device='cuda')
y_gpu = torch.tensor(y, device='cuda')
print(f"Data transferred to GPU. Shape: {X_gpu.shape}")
print(f"GPU memory now: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\nCheck nvidia-smi in another terminal to see GPU usage")
print("This script will pause for 10 seconds to keep GPU memory allocated")
time.sleep(10)  # Keep GPU memory allocated so it shows up in nvidia-smi

# Create some test data
X = np.random.random((1000, 10)).astype(np.float32)
y = np.random.randint(0, 2, 1000)

# Initialize models
print("Initializing cuSVC...")
model = cuSVC(probability=True)
print("Initializing cuTruncatedSVD...")
svd = cuTruncatedSVD(n_components=5)
print("Initializing cuStandardScaler...")
scaler = cuStandardScaler()

# Try to fit
print("Fitting SVD...")
X_svd = svd.fit_transform(X)
print("Fitting scaler...")
X_scaled = scaler.fit_transform(X_svd)
print("Fitting SVM...")
model.fit(X_scaled, y)

print("GPU test successful!")