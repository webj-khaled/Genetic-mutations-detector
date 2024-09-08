# Genetic-mutations-detector
This repository contains a machine learning model implemented using PyTorch. The model is designed to predict a continuous target variable based on various input features. The code provides flexibility for training, testing, and predicting using customizable neural network architectures and configurations.

Features
Model Training: Train the model using numerical and categorical features.
Prediction: Load pre-trained models to predict values for new datasets.
Cross-validation: K-fold cross-validation to improve model reliability.
Multiple Configurations: Supports various hyperparameters for tuning such as batch size, learning rates, number of layers, and activation functions.
Reproducibility: Optional flag for reproducibility, ensuring consistent results across runs.
Scalability: Supports different scaling techniques for data preprocessing.
Usage Examples
Training Mode:

bash
Copy code
python3 generic_nn_model_notes.py --feature_file ./SP1_features_corr2.tsv --num_feat_names ASA,size --cat_feat_names SecStr --target_name ddGexp --n_epochs 292 --hidden_layers 2 --num_nodes 24 --scalers RobustScaler --mode train
Prediction Mode:

bash
Copy code
python3 generic_nn_model_notes.py --mode predict --feature_file ./features_for_pred.csv --num_feat_names hydrophobicity,maxASA --model_file ./best_model.pt --pickled_scaler ./best_scaler.pkl --pickled_encoder ./best_encoder.pkl
Parameters:
--mode: Specify whether to train (train), test (test), or predict (predict).
--feature_file: Path to the file containing the features and target data.
--num_feat_names: Comma-separated list of numerical feature names.
--cat_feat_names: Comma-separated list of categorical feature names.
--n_epochs: Number of training epochs.
--scalers: Method for scaling the data (e.g., RobustScaler, StandardScaler).
--loss_fn: Loss function used for model training (e.g., SmoothL1Loss, MSELoss).
--activation_func: Activation functions to use in the neural network (e.g., ReLU, Sigmoid).
--model_file: Path to save or load the trained model for prediction.
