# Hybrid Quantum-Classical SVM for COVID-19 vs Pneumonia Classification

A novel machine learning approach combining IBM Quantum hardware with classical ensemble methods for automated COVID-19 vs Pneumonia classification from chest X-ray images.

## Results

| Metric | Value |
|--------|-------|
| Mean Accuracy | 94.13% ± 0.69% |
| Best Fold Accuracy | 95.33% |
| Mean F1-Score | 94.21% |
| Mean Precision | 94.16% |
| Mean Recall | 94.13% |
| Quantum Backend | IBM Torino (real quantum computer) |
| Quantum Circuits | 100 circuits executed |

## Key Features

- Real IBM Quantum hardware execution (ibm_torino)
- 5-fold stratified cross-validation
- Hybrid quantum-classical feature extraction
- Ensemble learning with 5 classifiers
- Comprehensive visualization and metrics
- Production-ready with model persistence

## Architecture

### Quantum Feature Extraction
- 5-layer parametrized quantum circuit
- 6 qubits with RY, RZ rotations and CNOT gates
- 512 shots per circuit
- Extracts 8 quantum features (superposition, entanglement, entropy, etc.)

### Classical Feature Extraction
- Statistical features (mean, std, percentiles, moments)
- Texture features (contrast, smoothness, gradients)
- Frequency domain features (FFT)
- 30+ engineered features total

### Ensemble Classifier
- SVM with RBF kernel (C=5000)
- SVM with Polynomial kernel (C=2000)
- Random Forest (300 trees)
- Gradient Boosting (150 estimators)
- Logistic Regression
- Soft voting with optimized weights [5, 4, 4, 3, 2]

## Dataset

- Training: 400 images (200 COVID-19 + 200 Pneumonia)
- Testing: 100 images (50 COVID-19 + 50 Pneumonia)
- Image size: 64×64 grayscale
- Data augmentation: 3× increase via Gaussian smoothing and contrast enhancement
