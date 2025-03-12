"""
SAE Error Probing AMI module for analyzing and probing sparse autoencoder activations.

This module provides tools for training and evaluating logistic regression probes
on various features derived from sparse autoencoder activations in transformer models.
"""

from .logistic_regression_probe import (
    # Feature Generation
    generate_probing_features,
    
    # Probe Training and Evaluation
    train_logistic_probe,
    evaluate_probe_full,
    get_probe_weights,
    train_and_evaluate_probe,
    
    # Feature Combination and Analysis
    combine_sae_and_error,
    compute_cosine_similarities,
    
    # Pipeline Functions
    run_probing_pipeline,
    process_dataset,
    
    # Utility Functions
    save_probes_and_compute_similarities
)

__all__ = [
    "generate_probing_features",
    "train_logistic_probe",
    "evaluate_probe_full",
    "get_probe_weights",
    "train_and_evaluate_probe",
    "combine_sae_and_error",
    "compute_cosine_similarities",
    "run_probing_pipeline",
    "process_dataset",
    "save_probes_and_compute_similarities",
]

__version__ = "0.1.0"