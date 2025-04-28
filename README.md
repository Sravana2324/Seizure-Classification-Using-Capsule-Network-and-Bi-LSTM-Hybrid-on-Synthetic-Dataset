# Seizure-Classification-Using-Capsule-Network-and-Bi-LSTM-Hybrid-on-Synthetic-Dataset

## Introduction

Seizure classification is a critical task in the field of neurological diagnostics, enabling timely and accurate treatment of epilepsy and other related disorders. This project presents a hybrid deep learning model that leverages Capsule Networks and Bidirectional Long Short-Term Memory (Bi-LSTM) networks to classify EEG data into three categories: normal, focal seizures, and generalized seizures. Notably, the dataset used for this project was synthetically generated to overcome the limitations of publicly available EEG datasets.

## Process

<ins>**Dataset Preparation**</ins>

A synthetic dataset was designed and generated to simulate realistic EEG recordings, accounting for three classes:

*  Normal: Baseline EEG without seizure activity.

*  Focal Seizures: Localized seizure activity affecting specific regions.

*  Generalized Seizures: Seizure activity spread across all regions.

Key steps in dataset generation included:

*  Baseline EEG Simulation: Using sine waves with alpha, beta, theta, and delta rhythms.

*  Noise and Artifacts: Adding realistic noise and common EEG distortions.

*  Seizure Patterns: Introducing focal and generalized seizure patterns.

*  Augmentation: Techniques such as random sampling, scaling, and temporal shifting were applied to enhance dataset diversity.

<ins>**Model Architecture**</ins>

The proposed hybrid model combines:

1.   Capsule Networks: Extract spatial hierarchies and connections.

2.   Bi-LSTM Networks: Analyze temporal dependencies in sequential EEG data.

The architecture includes:

*  Input Layer: Handles time-series EEG data.

*  Capsule Network: Extracts spatial features using convolutional layers, primary capsules, and dynamic routing.

*  Bi-LSTM Layer: Processes data bidirectionally to capture temporal relationships.

*  Fully Connected Layer: Combines spatial and temporal features.

*  Output Layer: Uses softmax activation for classification into three classes.

<ins>**Training and Evaluation**</ins>

*  Training: The model was trained using TensorFlow and Keras over 150 epochs with dynamic learning rate adjustment and early stopping.

*  Evaluation Metrics: Accuracy, precision, recall, F1-score, and confusion matrix were used for performance assessment.

## About the Model

*  Accuracy: The model achieved a test accuracy of 98.66% and a test loss of 0.0825.

*  Robustness: Balanced predictions across all classes, with low misclassification rates:
    -  Normal: 1.51%
    -  Focal: 1.70%
    -  General: 0.79%

*  Generalization: Training and validation curves show consistent performance without overfitting.

## Results

The hybrid model demonstrated exceptional classification performance:

*  Confusion Matrix: Accurate predictions for all three classes.

*  Classification Metrics: Precision, recall, and F1-scores were approximately 0.99 across all classes.

*  Visualizations: Training and validation accuracy and loss curves showed stable convergence.

## Conclusion

This project successfully developed a hybrid deep learning model for seizure classification using self-generated EEG datasets. By integrating Capsule Networks and Bi-LSTM, the model achieves superior accuracy and generalization, making it a potential tool for clinical applications. Future work includes validating the model on real-world EEG data and exploring advanced architectures to further improve performance.
