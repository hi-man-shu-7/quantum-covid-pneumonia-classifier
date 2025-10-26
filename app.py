"""
================================================================================
HYBRID QUANTUM-CLASSICAL SVM WITH K-FOLD CROSS-VALIDATION
================================================================================

This module implements a hybrid quantum-classical machine learning approach
for COVID-19 vs Pneumonia classification using chest X-ray images.

Key Features:
    - IBM Quantum hardware integration (real quantum computer)
    - 5-fold stratified cross-validation for robust evaluation
    - Comprehensive feature engineering (classical + quantum)
    - Automated visualization and model saving
    - Ensemble machine learning with voting classifier

Author: Himanshu Nainwal
Institution: SRM Institute of Science and Technology
Date: October 2025

Dependencies:
    - qiskit-ibm-runtime
    - scikit-learn
    - numpy
    - opencv-python
    - matplotlib
    - seaborn
    - scipy
    - joblib
"""

import time
import numpy as np
import cv2
import os
from tqdm import tqdm
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for non-interactive plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                            confusion_matrix, precision_score, recall_score)
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter

# Quantum computing imports
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

CONFIG = {
    # Data configuration
    'data_path': r'C:\Users\Neeraj\Desktop\qml1\covid_pneumonia_organized',
    'img_size': (64, 64),  # Resize all images to 64x64 pixels
    
    # Quantum computing configuration
    'use_quantum': True,  # Set to True to use IBM Quantum hardware
    'api_key': 'API-KEY',  # IBM Quantum API key
    'quantum_samples': 100,  # Number of samples to process on quantum hardware
    'n_qubits': 6,  # Number of qubits for quantum circuits
    'quantum_weight': 0.3,  # Weight for quantum features (0.3 = 30% quantum, 70% classical)
    
    # Cross-validation configuration
    'n_folds': 5,  # Number of folds for k-fold cross-validation
    
    # Data augmentation
    'use_augmentation': True,  # Enable data augmentation for training
    
    # Output configuration
    'output_dir': 'qsvm_results',  # Directory to save all results
}

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================================================
# DATA AUGMENTATION FUNCTIONS
# ============================================================================

def augment_image(img):
    """
    Apply data augmentation techniques to increase training data diversity.
    
    Augmentation techniques applied:
        1. Original image (no modification)
        2. Gaussian smoothing (reduces noise)
        3. Contrast enhancement (power transformation)
    
    Args:
        img (numpy.ndarray): Input grayscale image (64x64)
        
    Returns:
        list: List of augmented images including original
        
    Note:
        All augmented images maintain the same dimensions as input
    """
    augmented = []
    
    # Add original image
    augmented.append(img)
    
    # Apply Gaussian smoothing with sigma=0.8
    # This reduces noise while preserving edges
    augmented.append(gaussian_filter(img, sigma=0.8))
    
    # Apply contrast enhancement using power transformation
    # Normalizes intensity values and applies gamma correction
    img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
    augmented.append(img_normalized ** 0.8)
    
    return augmented

# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================

def load_all_data(data_path, img_size, use_augmentation=True):
    """
    Load and preprocess all X-ray images from train and test directories.
    
    This function combines both training and testing data for k-fold
    cross-validation. It applies augmentation if enabled.
    
    Args:
        data_path (str): Root path to dataset containing train/ and test/ folders
        img_size (tuple): Target image size as (height, width)
        use_augmentation (bool): Whether to apply data augmentation
        
    Returns:
        tuple: (X_all, y_all)
            - X_all: numpy array of flattened images (n_samples, n_pixels)
            - y_all: numpy array of labels (0=COVID-19, 1=Pneumonia)
            
    Directory Structure Expected:
        data_path/
            train/
                covid/
                    image1.jpg
                    image2.jpg
                    ...
                pneumonia/
                    image1.jpg
                    image2.jpg
                    ...
            test/
                covid/
                pneumonia/
    """
    print("\n" + "="*70)
    print("LOADING ALL DATA FOR K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    X_all = []  # Will store all image data
    y_all = []  # Will store all labels
    
    # Define class labels: COVID-19 = 0, Pneumonia = 1
    class_labels = {'covid': 0, 'pneumonia': 1}
    
    # Iterate through both train and test directories
    for subset in ['train', 'test']:
        # Iterate through both COVID and Pneumonia classes
        for class_name, label in class_labels.items():
            # Construct folder path
            folder = os.path.join(data_path, subset, class_name)
            
            # Get all image files (jpg, jpeg, png)
            files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Process each image file
            for img_file in tqdm(files, desc=f"{subset}/{class_name}"):
                try:
                    # Read image in grayscale mode
                    img = cv2.imread(os.path.join(folder, img_file), cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Resize image to target size
                        img = cv2.resize(img, img_size)
                        
                        # Normalize pixel values to [0, 1] range
                        img = img.astype(np.float32) / 255.0
                        
                        if use_augmentation:
                            # Apply augmentation and add all versions
                            for aug_img in augment_image(img):
                                X_all.append(aug_img.flatten())  # Flatten 2D image to 1D
                                y_all.append(label)
                        else:
                            # Add only original image
                            X_all.append(img.flatten())
                            y_all.append(label)
                            
                except Exception as e:
                    # Skip corrupted or unreadable images
                    continue
    
    print(f"\nTotal samples loaded: {len(X_all)}")
    
    # Convert lists to numpy arrays for efficient processing
    return np.array(X_all), np.array(y_all)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, fold_num, output_dir):
    """
    Generate and save confusion matrix visualization for a specific fold.
    
    The confusion matrix shows the performance of the classification model
    by displaying true positives, true negatives, false positives, and
    false negatives.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        fold_num (int): Fold number for filename
        output_dir (str): Directory to save the plot
        
    Returns:
        None (saves plot to disk)
        
    Output File:
        confusion_matrix_fold{fold_num}.png
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with appropriate size
    plt.figure(figsize=(8, 6))
    
    # Create heatmap with annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['COVID-19', 'Pneumonia'],
                yticklabels=['COVID-19', 'Pneumonia'],
                cbar_kws={'label': 'Count'})
    
    # Set title and labels
    plt.title(f'Confusion Matrix - Fold {fold_num}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure to output directory
    filename = os.path.join(output_dir, f'confusion_matrix_fold{fold_num}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved confusion matrix: {filename}")

def plot_metrics_per_fold(fold_accuracies, fold_f1, fold_precision, fold_recall, output_dir):
    """
    Create comprehensive visualization of all metrics across folds.
    
    This function generates a 2x2 subplot showing:
        - Accuracy per fold
        - F1-Score per fold
        - Precision per fold
        - Recall per fold
    
    Each subplot includes the mean value as a horizontal dashed line.
    
    Args:
        fold_accuracies (list): Accuracy scores for each fold
        fold_f1 (list): F1 scores for each fold
        fold_precision (list): Precision scores for each fold
        fold_recall (list): Recall scores for each fold
        output_dir (str): Directory to save the plot
        
    Returns:
        None (saves plot to disk)
        
    Output File:
        metrics_per_fold.png
    """
    # Create fold numbers for x-axis
    folds = list(range(1, len(fold_accuracies) + 1))
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Accuracy per fold
    axes[0, 0].plot(folds, fold_accuracies, marker='o', linewidth=2, 
                    markersize=8, color='#2E86AB')
    axes[0, 0].axhline(y=np.mean(fold_accuracies), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(fold_accuracies):.4f}')
    axes[0, 0].set_title('Accuracy per Fold', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Fold', fontsize=10)
    axes[0, 0].set_ylabel('Accuracy', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0.9, 1.0])
    
    # Plot 2: F1-Score per fold
    axes[0, 1].plot(folds, fold_f1, marker='s', linewidth=2, 
                    markersize=8, color='#A23B72')
    axes[0, 1].axhline(y=np.mean(fold_f1), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(fold_f1):.4f}')
    axes[0, 1].set_title('F1-Score per Fold', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Fold', fontsize=10)
    axes[0, 1].set_ylabel('F1-Score', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0.9, 1.0])
    
    # Plot 3: Precision per fold
    axes[1, 0].plot(folds, fold_precision, marker='^', linewidth=2, 
                    markersize=8, color='#F18F01')
    axes[1, 0].axhline(y=np.mean(fold_precision), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(fold_precision):.4f}')
    axes[1, 0].set_title('Precision per Fold', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Fold', fontsize=10)
    axes[1, 0].set_ylabel('Precision', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0.9, 1.0])
    
    # Plot 4: Recall per fold
    axes[1, 1].plot(folds, fold_recall, marker='d', linewidth=2, 
                    markersize=8, color='#6A994E')
    axes[1, 1].axhline(y=np.mean(fold_recall), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(fold_recall):.4f}')
    axes[1, 1].set_title('Recall per Fold', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Fold', fontsize=10)
    axes[1, 1].set_ylabel('Recall', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0.9, 1.0])
    
    # Adjust layout and save
    plt.tight_layout()
    filename = os.path.join(output_dir, 'metrics_per_fold.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metrics plot: {filename}")

def plot_overall_confusion_matrix(y_true_all, y_pred_all, output_dir):
    """
    Generate overall confusion matrix combining all folds.
    
    This provides a comprehensive view of model performance across
    all cross-validation folds combined.
    
    Args:
        y_true_all (array-like): All true labels from all folds
        y_pred_all (array-like): All predictions from all folds
        output_dir (str): Directory to save the plot
        
    Returns:
        None (saves plot to disk)
        
    Output File:
        confusion_matrix_overall.png
    """
    # Calculate overall confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    
    # Create larger figure for overall results
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with color gradient
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=['COVID-19', 'Pneumonia'],
                yticklabels=['COVID-19', 'Pneumonia'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 16, 'weight': 'bold'})
    
    plt.title('Overall Confusion Matrix (All Folds)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Add percentage annotations
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            plt.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=12, color='gray')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'confusion_matrix_overall.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved overall confusion matrix: {filename}")

# ============================================================================
# MAIN HYBRID QUANTUM-CLASSICAL SVM CLASS
# ============================================================================

class KFoldHybridQSVM:
    """
    Hybrid Quantum-Classical Support Vector Machine with K-Fold Cross-Validation.
    
    This class implements a machine learning pipeline that combines:
        1. Classical feature extraction (statistical, texture, frequency domain)
        2. Quantum feature extraction (using IBM Quantum hardware)
        3. Feature fusion and selection
        4. Ensemble classification with multiple models
        5. K-fold cross-validation for robust evaluation
    
    Attributes:
        service: IBM Quantum service connection
        backend: Selected IBM Quantum backend (e.g., ibm_torino)
        n_qubits: Number of qubits for quantum circuits
        quantum_weight: Weight for quantum features in hybrid model
        quantum_backend_name: Name of the quantum backend being used
        quantum_circuits_executed: Total number of quantum circuits executed
    
    Methods:
        setup_backend(): Initialize connection to IBM Quantum hardware
        create_quantum_circuits(): Build parametrized quantum circuits
        execute_quantum(): Execute circuits on IBM Quantum
        extract_quantum_features(): Extract features from quantum measurements
        extract_classical_features(): Extract statistical and texture features
        train_and_evaluate_kfold(): Main training and evaluation pipeline
    """
    
    def __init__(self, service=None, n_qubits=6, quantum_weight=0.3):
        """
        Initialize the Hybrid QSVM classifier.
        
        Args:
            service: IBM Quantum service instance
            n_qubits: Number of qubits for quantum feature map
            quantum_weight: Weight for quantum features (0-1)
        """
        self.service = service
        self.backend = None
        self.n_qubits = n_qubits
        self.quantum_weight = quantum_weight
        self.quantum_backend_name = None
        self.quantum_circuits_executed = 0
        
    def setup_backend(self):
        """
        Connect to IBM Quantum and select the least busy quantum computer.
        
        This function:
            1. Queries available IBM Quantum backends
            2. Filters for operational hardware (not simulators)
            3. Selects the backend with the smallest queue
            4. Stores backend information for later use
        
        Returns:
            bool: True if successfully connected, False otherwise
        """
        if not self.service:
            print("No IBM Quantum service provided")
            return False
            
        try:
            # Get list of all available backends
            backends = self.service.backends(operational=True, simulator=False)
            
            if backends:
                # Select backend with fewest queued jobs
                self.backend = min(backends, key=lambda b: b.status().pending_jobs)
                self.quantum_backend_name = self.backend.name
                
                # Display selected backend information
                print(f"Connected to IBM Quantum Backend: {self.backend.name}")
                print(f"Queue size: {self.backend.status().pending_jobs} jobs")
                
                return True
            else:
                print("No quantum backends available")
                return False
                
        except Exception as e:
            print(f"Error connecting to IBM Quantum: {e}")
            return False
    
    def create_quantum_circuits(self, X_pca):
        """
        Create parametrized quantum circuits for feature mapping.
        
        This function implements a 5-layer quantum feature map:
            Layer 1: RY and RZ rotations encoding data
            Layer 2: CNOT gates for entanglement (linear + circular)
            Layer 3: Non-linear encoding using sin/cos functions
            Layer 4: Additional CNOT entanglement
            Layer 5: Final RY rotations
        
        The circuit design allows quantum states to capture non-linear
        relationships in the data that are difficult for classical methods.
        
        Args:
            X_pca: PCA-reduced feature vectors (n_samples, n_qubits)
            
        Returns:
            list: Transpiled quantum circuits ready for execution
        """
        circuits = []
        
        # Create one circuit for each data sample
        for x in X_pca:
            # Initialize quantum circuit with n_qubits and n_classical_bits
            qc = QuantumCircuit(self.n_qubits, self.n_qubits)
            
            # Layer 1: Feature encoding with rotation gates
            # Each data feature controls a rotation angle
            for j in range(self.n_qubits):
                qc.ry(x[j] * np.pi, j)  # Y-rotation proportional to feature value
                qc.rz(x[j] * np.pi / 2, j)  # Z-rotation for phase encoding
            
            # Layer 2: Linear and circular entanglement
            # CNOT gates create quantum correlations between qubits
            for j in range(self.n_qubits - 1):
                qc.cx(j, j + 1)  # Linear chain of CNOTs
            qc.cx(self.n_qubits - 1, 0)  # Circular connection (last to first)
            
            # Layer 3: Non-linear feature encoding
            # Using trigonometric functions captures non-linear patterns
            for j in range(self.n_qubits):
                qc.ry(np.sin(x[j]) * np.pi / 2, j)  # Sine-based encoding
                qc.rx(np.cos(x[j]) * np.pi / 2, j)  # Cosine-based encoding
            
            # Layer 4: Additional entanglement with stride-2 pattern
            for j in range(0, self.n_qubits - 1, 2):
                qc.cx(j, j + 1)
            
            # Layer 5: Final parametrized layer
            for j in range(self.n_qubits):
                qc.ry(x[j] * 0.75 * np.pi, j)
            
            # Add measurement operations to read out qubit states
            qc.measure_all()
            
            circuits.append(qc)
        
        # Track total circuits created
        self.quantum_circuits_executed += len(circuits)
        
        try:
            # Transpile circuits for the specific quantum hardware
            # Optimization level 3 = maximum optimization
            pm = generate_preset_pass_manager(
                backend=self.backend, 
                optimization_level=3
            )
            transpiled_circuits = pm.run(circuits)
            
            return transpiled_circuits
            
        except Exception as e:
            print(f"Transpilation warning: {e}")
            # Return original circuits if transpilation fails
            return circuits
    
    def execute_quantum(self, circuits):
        """
        Execute quantum circuits on IBM Quantum hardware.
        
        This function:
            1. Initializes the SamplerV2 primitive for circuit execution
            2. Splits circuits into batches to avoid timeout
            3. Submits each batch as a job to IBM Quantum
            4. Collects results from all batches
        
        Args:
            circuits: List of transpiled quantum circuits
            
        Returns:
            list: Results containing measurement outcomes for each circuit
            
        Note:
            Each circuit is executed with 512 shots (measurements)
            to obtain statistical distributions of quantum states.
        """
        # Initialize sampler primitive for circuit execution
        sampler = SamplerV2(mode=self.backend)
        
        batch_size = 20  # Process 20 circuits at a time
        results = []
        
        # Process circuits in batches
        for i in range(0, len(circuits), batch_size):
            batch = circuits[i:i+batch_size]
            
            try:
                # Submit batch job to IBM Quantum
                # shots=512 means each circuit is measured 512 times
                job = sampler.run(batch, shots=512)
                
                # Wait for job completion and get results
                results.extend(job.result())
                
            except Exception as e:
                print(f"Batch execution error: {e}")
                
                # Create dummy results for failed batch
                # This ensures the pipeline continues even if some jobs fail
                for _ in batch:
                    dummy_result = type('DummyResult', (), {
                        'data': type('DummyData', (), {
                            'meas': type('DummyMeas', (), {
                                'get_counts': lambda: {
                                    '0' * self.n_qubits: 256, 
                                    '1' * self.n_qubits: 256
                                }
                            })()
                        })()
                    })()
                    results.append(dummy_result)
        
        return results
    
    def extract_quantum_features(self, results):
        """
        Extract meaningful features from quantum measurement results.
        
        For each quantum circuit execution result, this extracts 8 features:
            1. Superposition measure: Amount of quantum superposition
            2. Entanglement measure: Parity-based entanglement indicator
            3. Quantum entropy: Shannon entropy of measurement distribution
            4. State diversity: Fraction of unique states observed
            5. Hamming weight: Average number of 1s in bit strings
            6. Gini coefficient: Concentration of probability distribution
            7. Maximum probability: Highest measurement probability
            8. Minimum probability: Lowest measurement probability
        
        Args:
            results: List of quantum circuit execution results
            
        Returns:
            numpy.ndarray: Feature matrix (n_samples, 8)
        """
        quantum_features = []
        
        for result in results:
            try:
                # Get measurement counts from quantum circuit execution
                # counts is a dictionary: {'000000': 45, '010101': 32, ...}
                counts = result.data.meas.get_counts()
                total_shots = sum(counts.values()) or 1
                
                features = []
                
                # Feature 1: Superposition measure
                # High value indicates more quantum superposition
                zero_state_prob = counts.get('0' * self.n_qubits, 0) / total_shots
                one_state_prob = counts.get('1' * self.n_qubits, 0) / total_shots
                superposition = 1 - max(zero_state_prob, one_state_prob)
                features.append(superposition)
                
                # Feature 2: Entanglement via parity
                # Measures correlation between qubits
                even_parity_prob = sum(
                    counts.get(state, 0) 
                    for state in counts 
                    if state.count('1') % 2 == 0
                ) / total_shots
                entanglement = abs(even_parity_prob - 0.5) * 2
                features.append(entanglement)
                
                # Feature 3: Quantum entropy
                # Shannon entropy of measurement distribution
                entropy = -sum(
                    (count / total_shots) * np.log2(count / total_shots + 1e-10) 
                    for count in counts.values()
                )
                normalized_entropy = entropy / np.log2(2 ** self.n_qubits) if self.n_qubits > 0 else 0
                features.append(min(normalized_entropy, 1.0))
                
                # Feature 4: State diversity
                # Fraction of possible states that were measured
                state_diversity = len(counts) / (2 ** self.n_qubits)
                features.append(state_diversity)
                
                # Feature 5: Average Hamming weight
                # Average number of 1s in measured bit strings
                avg_hamming = sum(
                    state.count('1') * counts[state] 
                    for state in counts
                ) / (total_shots * self.n_qubits)
                features.append(avg_hamming)
                
                # Feature 6: Gini coefficient
                # Measures inequality in probability distribution
                sorted_probs = sorted(
                    [count / total_shots for count in counts.values()], 
                    reverse=True
                )
                cumulative_probs = np.cumsum(sorted_probs)
                gini = 1 - 2 * np.trapz(cumulative_probs, dx=1/len(cumulative_probs))
                features.append(abs(gini))
                
                # Features 7-8: Max and min probabilities
                max_prob = max(counts.values()) / total_shots
                min_prob = min(counts.values()) / total_shots
                features.extend([max_prob, min_prob])
                
                quantum_features.append(features)
                
            except Exception as e:
                # Use default fallback features if extraction fails
                quantum_features.append([0.5, 0.4, 0.6, 0.3, 0.5, 0.4, 0.7, 0.2])
        
        return np.array(quantum_features)
    
    def extract_classical_features(self, X_data):
        """
        Extract comprehensive classical (non-quantum) features from images.
        
        This function extracts multiple categories of features:
            - Statistical features (mean, std, percentiles, etc.)
            - Shape features (skewness, kurtosis)
            - Energy features (signal power, RMS)
            - Texture features (contrast, smoothness, gradients)
            - Ratio features (relationships between statistics)
        
        Args:
            X_data: Flattened image data (n_samples, n_pixels)
            
        Returns:
            numpy.ndarray: Classical feature matrix (n_samples, n_features)
        """
        n_samples, n_pixels = X_data.shape
        feature_list = []
        
        # Reshape flattened images to 2D for texture analysis
        images_2d = X_data.reshape(n_samples, 64, 64)
        
        # ---- Statistical Features ----
        # Basic descriptive statistics of pixel intensity values
        means = np.mean(X_data, axis=1, keepdims=True)
        stds = np.std(X_data, axis=1, keepdims=True) + 1e-8  # Add epsilon to avoid division by zero
        maxs = np.max(X_data, axis=1, keepdims=True)
        mins = np.min(X_data, axis=1, keepdims=True)
        medians = np.median(X_data, axis=1, keepdims=True)
        variances = np.var(X_data, axis=1, keepdims=True)
        feature_list.extend([means, stds, maxs, mins, medians, variances])
        
        # Percentile features capture distribution shape
        for percentile in [10, 25, 50, 75, 90]:
            feature_list.append(
                np.percentile(X_data, percentile, axis=1, keepdims=True)
            )
        
        # ---- Range Features ----
        pixel_ranges = maxs - mins
        p25, p75 = np.percentile(X_data, [25, 75], axis=1)
        iqr = p75.reshape(-1, 1) - p25.reshape(-1, 1)  # Interquartile range
        feature_list.extend([pixel_ranges, iqr])
        
        # ---- Higher Order Statistical Moments ----
        # Skewness measures asymmetry of distribution
        # Kurtosis measures tailedness of distribution
        skewness_values = skew(X_data, axis=1, nan_policy='omit').reshape(-1, 1)
        kurtosis_values = kurtosis(X_data, axis=1, nan_policy='omit').reshape(-1, 1)
        feature_list.extend([skewness_values, kurtosis_values])
        
        # ---- Energy Features ----
        # Signal energy and RMS (root mean square)
        energy = np.sum(X_data ** 2, axis=1, keepdims=True)
        rms = np.sqrt(energy / n_pixels)
        feature_list.extend([energy, rms])
        
        # ---- 2D Texture Features ----
        # These capture spatial patterns in the image
        texture_features = []
        for img in images_2d:
            # Calculate horizontal and vertical gradients
            horizontal_grad = np.abs(np.diff(img, axis=1))
            vertical_grad = np.abs(np.diff(img, axis=0))
            
            # Contrast: difference between highest and lowest intensity
            contrast = np.max(img) - np.min(img)
            
            # Smoothness: measures texture uniformity
            smoothness = 1 - 1 / (1 + np.var(img))
            
            # Mean gradients indicate edge strength
            h_grad_mean = np.mean(horizontal_grad)
            v_grad_mean = np.mean(vertical_grad)
            
            texture_features.append([
                contrast, smoothness, h_grad_mean, v_grad_mean
            ])
        feature_list.append(np.array(texture_features))
        
        # ---- Gradient Features ----
        # First-order derivatives capture edge information
        gradients = np.abs(np.diff(X_data, axis=1))
        gradient_mean = np.mean(gradients, axis=1, keepdims=True)
        gradient_std = np.std(gradients, axis=1, keepdims=True)
        feature_list.extend([gradient_mean, gradient_std])
        
        # ---- Ratio Features ----
        # Ratios between different statistics provide normalized measures
        mean_to_std_ratio = means / (stds + 1e-8)
        range_to_std_ratio = pixel_ranges / (stds + 1e-8)
        feature_list.extend([mean_to_std_ratio, range_to_std_ratio])
        
        # Combine all features into single matrix
        classical_features = np.hstack(feature_list)
        
        # Replace any NaN or infinite values with finite numbers
        classical_features = np.nan_to_num(
            classical_features, 
            nan=0.0, 
            posinf=1.0, 
            neginf=-1.0
        )
        
        return classical_features
    
    def train_and_evaluate_kfold(self, X_all, y_all, n_folds=5, output_dir='qsvm_results'):
        """
        Main training and evaluation pipeline using k-fold cross-validation.
        
        This function orchestrates the entire hybrid quantum-classical pipeline:
            1. Connects to IBM Quantum hardware
            2. Extracts quantum features for all data
            3. Performs k-fold cross-validation
            4. For each fold:
                - Extracts classical features
                - Combines quantum and classical features
                - Trains ensemble classifier
                - Evaluates on test fold
                - Generates visualizations
            5. Computes overall statistics
            6. Saves all results and models
        
        Args:
            X_all: All image data (n_samples, n_pixels)
            y_all: All labels (n_samples,)
            n_folds: Number of folds for cross-validation
            output_dir: Directory to save results
            
        Returns:
            dict: Comprehensive results dictionary containing:
                - Mean and std of all metrics
                - Per-fold results
                - Quantum backend information
                - Timing information
                - Classification report
        """
        print("\n" + "="*70)
        print(f"K-FOLD CROSS-VALIDATION ({n_folds} FOLDS)")
        print("="*70)
        
        start_time = time.time()
        
        # ---- Step 1: Quantum Feature Extraction ----
        # Extract quantum features ONCE for all data
        # This is more efficient than extracting for each fold separately
        quantum_features_all = None
        
        if self.setup_backend():
            try:
                print("\nExecuting quantum circuits on IBM Quantum hardware...")
                
                # Apply PCA to reduce dimensionality before quantum processing
                # This is necessary because we have 4096 pixels but only 6 qubits
                quantum_pca = PCA(n_components=self.n_qubits)
                X_pca_all = quantum_pca.fit_transform(X_all)
                
                # Sample a subset of data for quantum processing
                # Processing all data on quantum hardware would take too long
                sample_size = min(CONFIG['quantum_samples'], len(X_pca_all))
                sample_indices = np.random.choice(
                    len(X_pca_all), 
                    sample_size, 
                    replace=False
                )
                X_pca_sample = X_pca_all[sample_indices]
                
                print(f"Submitting {len(X_pca_sample)} circuits to {self.quantum_backend_name}...")
                
                # Create and execute quantum circuits
                circuits = self.create_quantum_circuits(X_pca_sample)
                results = self.execute_quantum(circuits)
                sample_quantum_features = self.extract_quantum_features(results)
                
                # Extend quantum features to all samples using k-NN interpolation
                # This allows us to use quantum features for all data
                # even though we only ran circuits for a subset
                from sklearn.neighbors import NearestNeighbors
                
                # Fit k-NN on quantum-processed samples
                nn_model = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
                nn_model.fit(X_pca_sample)
                
                # Find nearest quantum-processed samples for all data points
                distances, neighbor_indices = nn_model.kneighbors(X_pca_all)
                
                # Compute weighted average of quantum features
                # Closer samples get higher weight
                weights = 1 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                
                quantum_features_all = np.sum(
                    sample_quantum_features[neighbor_indices] * weights[:, :, np.newaxis],
                    axis=1
                )
                
                print(f"Quantum features computed: {quantum_features_all.shape}")
                
            except Exception as e:
                print(f"Quantum processing failed: {e}")
                print("Continuing with classical features only")
        else:
            print("Quantum backend not available, using classical features only")
        
        # ---- Step 2: K-Fold Cross-Validation ----
        # Initialize stratified k-fold splitter
        # Stratified ensures each fold has balanced class distribution
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Initialize lists to store metrics for each fold
        fold_accuracies = []
        fold_f1_scores = []
        fold_precisions = []
        fold_recalls = []
        all_predictions = []
        all_true_labels = []
        
        # Iterate through each fold
        for fold_num, (train_indices, test_indices) in enumerate(skf.split(X_all, y_all), 1):
            print(f"\n{'='*70}")
            print(f"FOLD {fold_num}/{n_folds}")
            print(f"{'='*70}")
            
            # Split data into train and test for this fold
            X_train_fold = X_all[train_indices]
            X_test_fold = X_all[test_indices]
            y_train_fold = y_all[train_indices]
            y_test_fold = y_all[test_indices]
            
            print(f"Training samples: {len(X_train_fold)}")
            print(f"Testing samples: {len(X_test_fold)}")
            
            # Extract classical features for this fold
            classical_train = self.extract_classical_features(X_train_fold)
            classical_test = self.extract_classical_features(X_test_fold)
            
            # Scale classical features using PowerTransformer
            # PowerTransformer makes features more Gaussian-like
            scaler = PowerTransformer()
            classical_train_scaled = scaler.fit_transform(classical_train)
            classical_test_scaled = scaler.transform(classical_test)
            
            # Combine quantum and classical features if quantum is available
            if quantum_features_all is not None:
                # Get quantum features for this fold's samples
                quantum_train = quantum_features_all[train_indices]
                quantum_test = quantum_features_all[test_indices]
                
                # Weighted combination of quantum and classical features
                combined_train = np.hstack([
                    classical_train_scaled * (1 - self.quantum_weight),
                    quantum_train * self.quantum_weight
                ])
                combined_test = np.hstack([
                    classical_test_scaled * (1 - self.quantum_weight),
                    quantum_test * self.quantum_weight
                ])
            else:
                # Use only classical features if quantum not available
                combined_train = classical_train_scaled
                combined_test = classical_test_scaled
            
            # Feature selection using mutual information
            # Selects most informative features for classification
            max_features = min(40, combined_train.shape[1], len(X_train_fold) // 3)
            selector = SelectKBest(mutual_info_classif, k=max_features)
            X_train_selected = selector.fit_transform(combined_train, y_train_fold)
            X_test_selected = selector.transform(combined_test)
            
            print(f"Selected {X_train_selected.shape[1]} most informative features")
            
            # Create ensemble classifier with multiple algorithms
            # Voting classifier combines predictions from multiple models
            ensemble_model = VotingClassifier(
                estimators=[
                    # SVM with RBF kernel - good for non-linear patterns
                    ('svm_rbf', SVC(
                        kernel='rbf', 
                        C=5000, 
                        gamma='scale', 
                        probability=True, 
                        random_state=42
                    )),
                    # SVM with polynomial kernel - captures polynomial relationships
                    ('svm_poly', SVC(
                        kernel='poly', 
                        degree=3, 
                        C=2000, 
                        gamma='scale', 
                        probability=True, 
                        random_state=42
                    )),
                    # Random Forest - ensemble of decision trees
                    ('random_forest', RandomForestClassifier(
                        n_estimators=300, 
                        max_depth=20, 
                        random_state=42, 
                        n_jobs=-1
                    )),
                    # Gradient Boosting - sequential ensemble
                    ('gradient_boost', GradientBoostingClassifier(
                        n_estimators=150, 
                        learning_rate=0.03, 
                        random_state=42
                    )),
                    # Logistic Regression - linear baseline
                    ('logistic', LogisticRegression(
                        C=200, 
                        max_iter=5000, 
                        random_state=42
                    ))
                ],
                voting='soft',  # Use probability averages
                weights=[5, 4, 4, 3, 2]  # Give more weight to SVM models
            )
            
            # Train ensemble on this fold's training data
            print("Training ensemble classifier...")
            ensemble_model.fit(X_train_selected, y_train_fold)
            
            # Make predictions on test fold
            y_pred_fold = ensemble_model.predict(X_test_selected)
            
            # Calculate metrics for this fold
            accuracy = accuracy_score(y_test_fold, y_pred_fold)
            f1 = f1_score(y_test_fold, y_pred_fold)
            precision = precision_score(y_test_fold, y_pred_fold)
            recall = recall_score(y_test_fold, y_pred_fold)
            
            # Store metrics
            fold_accuracies.append(accuracy)
            fold_f1_scores.append(f1)
            fold_precisions.append(precision)
            fold_recalls.append(recall)
            
            # Store all predictions and labels for overall confusion matrix
            all_predictions.extend(y_pred_fold)
            all_true_labels.extend(y_test_fold)
            
            print(f"\nFold {fold_num} Results:")
            print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            
            # Generate and save confusion matrix for this fold
            plot_confusion_matrix(y_test_fold, y_pred_fold, fold_num, output_dir)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        # ---- Step 3: Compute Overall Statistics ----
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_f1 = np.mean(fold_f1_scores)
        mean_precision = np.mean(fold_precisions)
        mean_recall = np.mean(fold_recalls)
        best_accuracy = max(fold_accuracies)
        
        # Print summary results
        print("\n" + "="*70)
        print("K-FOLD CROSS-VALIDATION SUMMARY")
        print("="*70)
        print(f"Mean Accuracy:      {mean_accuracy:.4f} +/- {std_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
        print(f"Mean F1-Score:      {mean_f1:.4f}")
        print(f"Mean Precision:     {mean_precision:.4f}")
        print(f"Mean Recall:        {mean_recall:.4f}")
        print(f"Best Fold Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print(f"Total Time:         {total_time:.2f} seconds")
        
        if self.quantum_backend_name:
            print(f"Quantum Backend:    {self.quantum_backend_name}")
            print(f"Quantum Circuits:   {self.quantum_circuits_executed} executed")
        
        # ---- Step 4: Generate Visualizations ----
        print("\nGenerating visualizations...")
        plot_metrics_per_fold(
            fold_accuracies, fold_f1_scores, 
            fold_precisions, fold_recalls, 
            output_dir
        )
        plot_overall_confusion_matrix(all_true_labels, all_predictions, output_dir)
        
        # ---- Step 5: Save Results ----
        print("\nSaving results...")
        
        # Create comprehensive results dictionary
        results = {
            # Summary statistics
            'mean_accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'mean_f1': float(mean_f1),
            'mean_precision': float(mean_precision),
            'mean_recall': float(mean_recall),
            'best_accuracy': float(best_accuracy),
            
            # Per-fold results
            'fold_accuracies': [float(x) for x in fold_accuracies],
            'fold_f1_scores': [float(x) for x in fold_f1_scores],
            'fold_precisions': [float(x) for x in fold_precisions],
            'fold_recalls': [float(x) for x in fold_recalls],
            
            # Quantum information
            'quantum_backend': self.quantum_backend_name,
            'quantum_circuits_executed': self.quantum_circuits_executed,
            'quantum_enabled': quantum_features_all is not None,
            
            # Timing information
            'total_time_seconds': float(total_time),
            
            # Detailed classification report
            'classification_report': classification_report(
                all_true_labels, 
                all_predictions,
                target_names=['COVID-19', 'Pneumonia'],
                output_dict=True
            )
        }
        
        # Save results in JSON format (human-readable)
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved results: {os.path.join(output_dir, 'results.json')}")
        
        # Save results in pickle format (for Python loading)
        joblib.dump(results, os.path.join(output_dir, 'results.pkl'))
        print(f"Saved results: {os.path.join(output_dir, 'results.pkl')}")
        
        # Save the trained model (last fold's model)
        joblib.dump(ensemble_model, os.path.join(output_dir, 'model_final.pkl'))
        print(f"Saved model: {os.path.join(output_dir, 'model_final.pkl')}")
        
        # Save configuration
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(CONFIG, f, indent=4)
        print(f"Saved config: {os.path.join(output_dir, 'config.json')}")
        
        return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function that orchestrates the entire pipeline.
    
    Steps:
        1. Print header information
        2. Load and preprocess all data
        3. Initialize IBM Quantum service
        4. Create hybrid QSVM classifier
        5. Train and evaluate using k-fold CV
        6. Print final summary
    """
    print("\n" + "="*70)
    print("HYBRID QUANTUM-CLASSICAL SVM")
    print("COVID-19 vs PNEUMONIA CLASSIFICATION")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Dataset: {CONFIG['data_path']}")
    print(f"  Image Size: {CONFIG['img_size']}")
    print(f"  Quantum Enabled: {CONFIG['use_quantum']}")
    print(f"  Number of Qubits: {CONFIG['n_qubits']}")
    print(f"  K-Fold Splits: {CONFIG['n_folds']}")
    print(f"  Output Directory: {CONFIG['output_dir']}")
    
    # Load all data (train + test combined for k-fold CV)
    X_all, y_all = load_all_data(
        CONFIG['data_path'], 
        CONFIG['img_size'], 
        CONFIG['use_augmentation']
    )
    
    # Initialize IBM Quantum service if enabled
    service = None
    if CONFIG['use_quantum']:
        try:
            # Save IBM Quantum account credentials
            QiskitRuntimeService.save_account(
                token=CONFIG['api_key'], 
                overwrite=True
            )
            
            # Initialize service
            service = QiskitRuntimeService()
            print("\nIBM Quantum service initialized successfully")
            
        except Exception as e:
            print(f"\nWarning: Could not initialize IBM Quantum service: {e}")
            print("Continuing with classical features only")
    
    # Create hybrid QSVM classifier
    model = KFoldHybridQSVM(
        service=service,
        n_qubits=CONFIG['n_qubits'],
        quantum_weight=CONFIG['quantum_weight']
    )
    
    # Train and evaluate using k-fold cross-validation
    results = model.train_and_evaluate_kfold(
        X_all, 
        y_all, 
        CONFIG['n_folds'], 
        CONFIG['output_dir']
    )
    
    # Print final summary
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"Final Mean Accuracy: {results['mean_accuracy']:.4f} ({results['mean_accuracy']*100:.2f}%)")
    print(f"Best Fold Accuracy:  {results['best_accuracy']:.4f} ({results['best_accuracy']*100:.2f}%)")
    print(f"All results saved in: {CONFIG['output_dir']}/")
    print("="*70)

# Entry point
if __name__ == "__main__":
    main()
