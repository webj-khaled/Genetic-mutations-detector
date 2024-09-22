usage=\
'''
This file contains code for training and evaluating a machine learning model using PyTorch. 
The model is designed to predict a continuous target variable based on input features.

Note that comma separated lists of values may not contains spaces! 

## Usage examples

python3 generic_nn_model_notes.py python3 generic_nn_model_notes.py --feature_file ./SP1_features_corr2.tsv --num_feat_names ASA,size,monomers,pH,dpSFFCB,dpSFFCA,dcontSFF,dmass,diso,dhydro --cat_feat_names SecStr --encoding onehot --target_name ddGexp --n_epochs 292 --hidden_layers 2 --num_nodes 24 --scalers RobustScaler --loss_fn SmoothL1Loss --learning_rates 0.0001 --batch_sizes 128 --repro --mode train --activation_func ReLU

python3 generic_nn_model.py --mode predict --feature_file ./features_for_pred.csv --num_feat_names hydrophobicity,maxASA,cont_CA_6,cont_CA_7,cont_CA_8,cont_CA_9,cont_CA_10,cont_CA_11,cont_CA_12,cont_CB_6,cont_CB_7,cont_CB_8,cont_CB_9,cont_CB_10,cont_CB_11,cont_CB_12 --cat_feat_names oneLetterCode,SS --pickled_scaler ./best_scaler.pkl --pickled_encoder ./best_encoder.pkl --model_file ./best_model.pt

### Parameters 

--mode   :   This argument specifies the mode of the script; training ('train'), evaluating ('test') or predicting ('predict'). 
            Default value is 'train'.
--feature_file  :    Specify the path to the input .csv file containing the feature and target columns. Columns need to
                     be tab separated. A single header row defines the choices for feature and target names.
                    Feature file should be tab-seperated and header is required. Default value is './features351.csv'.
--encoding   :   Specify the encoding used for categorical data; Ordinal ('ordinal') or One Hot Encoding ('onehot'). 
                Default value is 'onehot'.
--num_feat_names :   Specify by title the numerical feature columns used for training.
--cat_feat_names :   Specify by title the categorical feature columns used for training.
--target_name    :   Specify by title the the target column used for training.
--batch_sizes*    :   Define the batch size for training loop. Predefined options [16, 32, 64, 128, 256]. 
--learning_rates* :   Set the learning rate for the optimizer. Predefined options [0.0001, 0.001, 0.01]. 
--n_folds    :   Define the number of folds for sklearn kfold cross validation. Default value is 5.
--n_epochs*   :   Specify the number of epochs for training the model.  Predefined options [50, 100, 150, 200, 300, 500]. 
--hidden_layers*  :   Specify the number of hidden layers in the neural network. Predefined options [1, 2, 3].
--num_nodes*  :   Set the number of nodes in the hidden layers of the neural network. Predefined options [16, 24, 32].
--scalers*    :   Specify the method/technique for scaling the data. Predefined options [RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer].
--loss_fn_arr*    :   Select the loss function for model training. Predefined options [SmoothL1Loss, L1Loss, MSELoss].
--model_file     :   Specify the path to the saved model used for prediction.
--pickled_scaler :   Specify the path to the saved scaler used for prediction.
--pickled_encoder    :   Specify the path to the saved encoder used for prediction.
--repro              : set seed and configurations for reproducability
--activation_func    : choose activation funftions to loop through [ 'ReLU','Sigmoid','Tanh','LeakyReLU' 'ELU' ,'SELU', 'Softplus'
'Softsign']
*If not specified as a command line argument, the model will be trained for each one of the predefined options.
User can also train the model using custom option, not existing in the predefined ones.

Author: kHALED ADEL

'''



import sys

#check if usage required
if len(sys.argv) < 2 or (len(sys.argv)==2 and sys.argv[1] in ['-h','-help','--help']):
    print(usage)
    sys.exit()

import argparse


# Create argument parser object
parser = argparse.ArgumentParser()

#NOTE: the argpare.Action concept did not deliver default values -> replaced by get_instances()


# Helper function for argument parser to be able to handle and parse comma-separated integers
def comma_separated_ints(value):
    """
    Parse a comma-separated string of integers.

    Args:
        value (str): The comma-separated string of integers.

    Returns:
        list of int: A list of parsed integers.

    Raises:
        argparse.ArgumentTypeError: If the input string cannot be converted to a list of integers.
    """

    try:
        return [int(item) for item in value.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid int value: {value}") from e

# Helper function for argument parser to be able to handle and parse comma-separated floats
def comma_separated_floats(value):
    """
    Parse a comma-separated string of floats.

    Args:
        value (str): The comma-separated string of floats.

    Returns:
        list of float: A list of parsed floats.

    Raises:
        argparse.ArgumentTypeError: If the input string cannot be converted to a list of floats.
    """
    
    try:
        return [float(item) for item in value.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}") from e


def get_instances(oname,choices,idict):
    """Parse a comma-separated string naming classes and return a
       list instances thereof
    
    Args:
        oname (str): option name (just for error message)
        choices (str): A list of class names
        idict (dict): mapping of a class name to an instance of the desired class
                      { 'name1': name1() ....}
    Returns:
        list of instances
   
    """
    r=[]
    for n in choices.split(','):
        n=n.strip()
        if n in idict: r.append(idict[n])
        else:
            print(n,"is not a choice for",oname)
            print(n," choices are: \n", ','.join(list(idict.keys())))
            print(usage)
            sys.exit()
    return r

# Add arguments to the parser
parser.add_argument('--mode', type=str, choices=['train','test','predict'], default='train', help='Script mode')
parser.add_argument('--feature_file', type=str, default='./features351.csv', help='Path to the feature file')
parser.add_argument('--num_feat_names', type=str, default='hydrophobicity,maxASA,cont_CA_6,cont_CA_7,cont_CA_8,cont_CA_9,cont_CA_10,cont_CA_11,cont_CA_12,cont_CB_6,cont_CB_7,cont_CB_8,cont_CB_9,cont_CB_10,cont_CB_11,cont_CB_12', help='Comma-separated list of numerical feature names')
parser.add_argument('--cat_feat_names', type=str, default='oneLetterCode,SS', help='Comma-separated list of categorical feature names')
parser.add_argument('--target_name', type=str, default='ASA', help='Target variable name')
parser.add_argument('--encoding', type=str, choices=['onehot','ordinal'],default='onehot', help='Encoding method (ordinal/onehot)')
parser.add_argument('--batch_sizes', type=comma_separated_ints, default=[16, 32, 64, 128, 256], help='Batch size')
parser.add_argument('--learning_rates', type=comma_separated_floats, default=[0.0001, 0.001, 0.01], help='Learning rate')
parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for kfold cross validation')
parser.add_argument('--n_epochs', type=comma_separated_ints, default=[50, 100, 150, 200, 300, 500], help='Number of epochs')
parser.add_argument('--hidden_layers', type=comma_separated_ints, default=[1, 2, 3], help='Number of hidden layers')
parser.add_argument('--num_nodes', type=comma_separated_ints, default=[16, 24, 32], help='Number of nodes in each hidden layer')
parser.add_argument('--scalers', type=str, default='RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer', help='Scaler function')#, action=ScalersAction)
parser.add_argument('--loss_fn', type=str, default='SmoothL1Loss, L1Loss, MSELoss', help='Loss function')#, action=LossFunctionsAction)
parser.add_argument('--model_file', type=str, default='./best_model.pt', help='Path to the trained model file')
parser.add_argument('--pickled_scaler', type=str, default='./best_scaler.pkl', help='Path to the pickled scaler file')
parser.add_argument('--pickled_encoder', type=str, default='./best_encoder.pkl', help='Path to the pickled encoding file')
parser.add_argument('--repro', action='store_true', help='Set seeds and configurations for reproducibility')
parser.add_argument('--activation_func', type=str, default='ReLU,Sigmoid,Tanh,LeakyReLU,ELU,SELU,Softplus,Softsign', help='Comma-separated list of activation functions')

# Parse the arguments
args = parser.parse_args()

##################################################
#
# main code starts here
#
##################################################
import itertools
import os
import tempfile
import csv
import math
import time
import numpy as np
import pandas as pd
from random import randint, seed as py_seed
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from pickle import dump, load
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer

if args.repro:
    # minimizing randomness in our results
    # Set seeds for reproducibility
    random_seed = 42
    py_seed(random_seed) #ets the seed for Python's built-in random module.
    np.random.seed(random_seed) #This line sets the seed for NumPy's random number generator. 
    torch.manual_seed(random_seed) #Sets the seed for generating random numbers on all devices. It ensures that any random operations performed will produce consistent results.
    #torch.cuda.manual_seed_all(random_seed) 
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
    
    # Set number of threads for script execution
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

#dicts containing option-name - object-instance mapping used by get_instances()

scalers_dict = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'PowerTransformer': PowerTransformer()
}

loss_functions_dict = {
    'SmoothL1Loss': nn.SmoothL1Loss(),
    'L1Loss' : nn.L1Loss(),
    'MSELoss': nn.MSELoss(),
    
}

activation_functions_dict = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'LeakyReLU': nn.LeakyReLU(),
    'ELU': nn.ELU(),
    'SELU': nn.SELU(),
    'Softplus': nn.Softplus(),
    'Softsign': nn.Softsign(),
}


# Extract parsed arguments
mode = args.mode
feature_file = args.feature_file
num_feat_names = list(args.num_feat_names.split(','))
cat_feat_names = list(args.cat_feat_names.split(','))
target_name = args.target_name
encoding_method = args.encoding #list(args.encoding.split(','))
batch_sizes = args.batch_sizes
learning_rates = args.learning_rates
n_folds = args.n_folds
n_epochs = args.n_epochs
hidden_layers = args.hidden_layers
num_nodes = args.num_nodes

scalers = get_instances('scalers',args.scalers,scalers_dict)
loss_fn_arr = get_instances('loss_fn',args.loss_fn,loss_functions_dict)
activation_func = get_instances('activation_func',args.activation_func,activation_functions_dict)
model_file = args.model_file
pickled_scaler = args.pickled_scaler
pickled_encoder = args.pickled_encoder

# Read feature file specified from the arguments
data = pd.read_csv(feature_file, delimiter="\t")

# Store initial columns from feature file
initial_feat_file_cols = data.columns.to_list()

# Get numerical feature columns specified in arguments
def get_num_data_df(data, num_feat_names):
    """
    Extracts numerical feature columns from the data DataFrame based on specified column names.

    Args:
        data (DataFrame): The DataFrame containing the data.
        num_feat_names (list): A list of numerical feature column names.

    Returns:
        DataFrame: A DataFrame containing only the specified numerical feature columns.
    """
    
    if len(num_feat_names) > 0 :

        missing_columns = [col for col in num_feat_names if col not in data.columns]

        if missing_columns:
            print("Numerical columns missing in data file:", missing_columns)
        else:
            # Proceed with selecting columns for numerical features
            num_data = data[num_feat_names]
    else: 
        num_data = pd.DataFrame()

    return num_data

# Get raw categorical feature columns specified in arguments
def get_raw_cat_fata_df(data, cat_feat_names):
    """
    Extracts raw categorical feature columns from the data DataFrame based on specified column names.

    Args:
        data (DataFrame): The DataFrame containing the data.
        cat_feat_names (list): A list of raw categorical feature column names.

    Returns:
        DataFrame: A DataFrame containing only the specified raw categorical feature columns.
    """

    if len(cat_feat_names) > 0 :

        missing_columns = [col for col in cat_feat_names if col not in data.columns]

        if missing_columns:
            print("Categorical columns missing in data file:", missing_columns)
        else:
            # Proceed with selecting columns for numerical features
            raw_cat_data = data[cat_feat_names]
    else: 
        raw_cat_data = pd.DataFrame()

    return raw_cat_data

# Get encoded categorical feature columns specified in arguments
def get_encoded_cat_data_df(data, cat_feat_names, encoding):
    """
    Extracts encoded categorical feature columns from the data DataFrame based on specified column names.

    Args:
        data (DataFrame): The DataFrame containing the data.
        cat_feat_names (list): A list of categorical feature column names.

    Returns:
        DataFrame: A DataFrame containing encoded categorical features.
        dict: A dictionary mapping each categorical feature column to its ordinal encoding.
        dict: A dictionary mapping each categorical feature column to its one-hot encoding.
        OneHotEncoder: The fitted OneHotEncoder object.
    """

    # Initialize empty ordinal and one-hot mappings
    ordinal_mappings = {}
    onehot_mappings = {}
    
    if len(cat_feat_names) > 0 :

        # Define Ordinal Encoding for catecorical values
        if encoding == "ordinal":
            encoder = OrdinalEncoder()
            
            # Fit and transform categorical features
            cat_data_encoded = encoder.fit_transform(data[cat_feat_names])

            # Create ordinal mappings for each categorical feature
            for i, feature in enumerate(cat_feat_names):
                ordinal_mappings[feature] = {label: index for index, label in enumerate(encoder.categories_[i])}

            # Create DataFrame for encoded categorical features
            cat_data = pd.DataFrame(cat_data_encoded, columns=cat_feat_names)

        # Define One Hot Encoding for catecorical values
        elif encoding == "onehot":
            encoder = OneHotEncoder(sparse_output=False)

            # Fit and transform categorical features
            cat_data_encoded = encoder.fit_transform(data[cat_feat_names])
                
            # Convert the encoded data into a DataFrame
            cat_data = pd.DataFrame(cat_data_encoded, columns=encoder.get_feature_names_out(cat_feat_names))

            # Get the one-hot encoded feature names
            one_hot_feature_names = encoder.get_feature_names_out(cat_feat_names)

            #@TODO check for more save solution is column name is part of another one (eg mutAA mut)
            # Create one-hot mappings for each categorical feature
            for fn in cat_feat_names: onehot_mappings[fn]=[]
            for n in one_hot_feature_names:
                for fn in cat_feat_names:
                    if n.startswith(fn+"_"): onehot_mappings[fn].append(n.replace(fn+"_",""))

    else:
        # If no categorical feature titles were specified, creaty empty categorical features dataframe
        cat_data = pd.DataFrame()
    
    return cat_data, ordinal_mappings, onehot_mappings, encoder


# Create folder to store best parameters training loss and results of script
def create_training_folder(folder_prefix="training_"):
    """
    Creates folder to store the results of the script execution.

    Args:
        folder_prefix: The prefix of the folder name to be created.
    """
    return tempfile.mkdtemp(prefix=folder_prefix,dir=os.getcwd())


def check_scaler_for_preprocessing(scaler, file):
    """
    Writes information about the scaler used for preprocessing into a file.

    Args:
        scaler: The scaler object used for preprocessing.
        file: The file object to write the information into.
    """

    # Check if the scaler is a RobustScaler instance
    if isinstance(scaler, RobustScaler):
        # Write necessary components for later C++ use of RobustScaler on file
        file.write("\nSCALE:")
        file.write(str(list(scaler.scale_)))
        file.write("\nCENTER:")
        file.write(str(list(scaler.center_)))
        file.write("\nFORMULA: x_scaled = (x_init-center) / scale\n")

    # Check if the scaler is a StandardScaler instance
    elif isinstance(scaler, StandardScaler):
        # Write necessary components for later C++ use of StandardScaler on file
        file.write("\nSCALE: ")
        file.write(str(list(scaler.scale_)))
        file.write("\nMEAN: ")
        file.write(str(list(scaler.mean_)))
        file.write("\nFORMULA: x_scaled = (x_init-mean) / scale")

    # Check if the scaler is a MinMaxScaler instance
    elif isinstance(scaler, MinMaxScaler):
        # Write necessary components for later C++ use of MinMaxScaler on file
        file.write("\nSCALE: ")
        file.write(str(list(scaler.scale_)))
        file.write("\nMIN: ")
        file.write(str(list(scaler.min_)))
        file.write("\nFORMULA: x_scaled = (x_init * scale) + min\n")

    # Check if the scaler is a PowerTransformer instance
    elif isinstance(scaler, PowerTransformer):
        # Write necessary components for later C++ use of PowerTransformer on file
        file.write("\nLAMBDAS: ")
        file.write(str(list(scaler.lambdas_)))
        file.write("\nMEAN: ")
        file.write(str(list(X_train_mean)))
        file.write("\nSTD: ")
        file.write(str(list(X_train_std)))
        file.write("\nFORMULA: x_scaled = (x_transformed - mean) / std")
        file.write("\n*Mean and STD refer to the transformed values.\n")

def reset_weights(m):
    """
    Resets the weights of trainable parameters in the given model to avoid weight spillage.

    Args:
        m: The model whose weights need to be reset.
    """

    # Iterate through all the layers in the model
    for layer in m.children():
        # Check if the layer has the method 'reset_parameters'
        if hasattr(layer, 'reset_parameters'):
            # Print a message indicating the layer whose parameters are being reset
            print(f'Reset trainable parameters of layer = {layer}')
            # Reset the parameters of the layer
            layer.reset_parameters()

# Extract numerical feature columns from the "data" DataFrame
num_data = get_num_data_df(data, num_feat_names)

# Extract raw categorical feature columns from the "data" DataFrame
cat_data_raw = get_raw_cat_fata_df(data, cat_feat_names)

# Concatenate raw numerical and categorical features into a single DataFrame
all_features_raw = pd.concat([num_data, cat_data_raw], axis=1)

# Get the titles of all raw features
all_raw_features_titles = all_features_raw.columns.to_list()

#helper class for best loss solution
class BestLossData:
    def __init__(self):
        self.loss_val = np.inf   # init to infinity
        self.weights = None
        self.epoch = 0
        self.loss_fn = None
        self.optimizer = None
        self.scaler = None
        self.batch_size = None
        self.lr= None
        self.n_epochs = None
        self.hidden_layers = None
        self.num_nodes = None
        self.val_pred_values = []
        self.val_truth_values = []
        self.fold = None
        self.activation = None
        
    def print_vals(self,of=sys.stdout):
        # Save best hyper-parameters on text file
        print(f"Best loss fn: {self.loss_fn}",file=of)
        print(f"Best optimizer: {self.optimizer}",file=of)
        print(f"Best epoch: {self.epoch+1}",file=of)
        print("Loss: %.2f" % self.loss_val,file=of)
        print(f"Best scaler: {self.scaler}", file=of)
        print(f"Batch size used: {self.batch_size}",file=of)
        print(f"Best lr: {self.lr}",file=of)
        print(f"Number of epochs: {self.n_epochs}",file=of)
        print(f"Network hidden layers: {self.hidden_layers}",file=of) 
        print(f"Network nodes: {self.num_nodes}",file=of)
        print(f"Best fold: {self.fold}", file=of)
        print(f"Best activation function: {self.activation}", file=of) 

    def save_model(self,name='best_model.pt'):
        print("TBI")

    def pkl_scaler(self,name='best_scaler.pkl'):
        print("TBI")

    def pkl_encoder(self,name='best_encoder.pkl'):
        print("TBI")

class PerParamterLoss:
    def __init__(self,outfile=None):
        if outfile==None: self.of = sys.stdout
        else: self.of = open(outfile,'w')

    def write_header(self,hstr):
        h=[]
        for p in hstr.split(','):
            h.append(p)
        h.append("loss")
        self.of.write('\t'.join(h)+'\n')
            
    def write_vals(self,loss,*args):
        v=[]
        for p in args:
            v.append(p.__str__().replace("\n"," ").split()[0])
        v.append(loss.__str__())
        self.of.write('\t'.join(v)+'\n')
    
        
############################################    PREDICTION MODE    ############################################

if mode == 'predict':

    # Load the scaler from the pickle file
    scaler= load(open(pickled_scaler, 'rb'))
    print(type(scaler)) 

    # Load the encoding method from the pickle file
    encoder= load(open(pickled_encoder, 'rb'))
    print(type(encoder))

    # Load the trained model
    model = torch.jit.load(model_file)

    # Extract raw categorical feature columns from the "data" DataFrame
    cat_data_raw = data[cat_feat_names]

    # Transform raw categorical features using the encoder
    enc_cat_data = encoder.transform(cat_data_raw)
    
    # Create a DataFrame for encoded categorical features
    cat_data = pd.DataFrame(enc_cat_data, columns=encoder.get_feature_names_out(cat_feat_names))

    # Concatenate numerical and encoded categorical data
    all_feat_data = pd.concat([num_data, cat_data], axis=1)
    
    # Get the values of all features
    X_pred_raw = all_feat_data.values

    # Preprocess the data using the scaler
    X_pred_scaled = scaler.transform(X_pred_raw)
    
    # Convert the preprocessed data to a 2D PyTorch tensor
    X_pred_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32)

    # Set the model to evaluation mode and perform inference without gradient tracking
    with torch.no_grad():
        model.eval()
        # Predict using the model
        y_pred = model(X_pred_tensor)

    # Convert the predictions from PyTorch tensor to a numpy array (otherwise cannot be appended to dataframe)
    predictions = y_pred.numpy()

    # Add the predictions as a new column to the "data" DataFrame
    data['prediction'] = predictions

    # Save the modified DataFrame to the feature CSV file
    data.to_csv(feature_file, sep='\t', index=False)

    # Exit the script
    sys.exit()

###############################################################################################################  

# If in plotting/testing mode ('test'), define specific parameters for use in training cycle
if mode == 'test':

    scalers = [scalers[0]]
    loss_fn_arr = [loss_fn_arr[0]]
    learning_rates = [learning_rates[0]]
    batch_sizes = [batch_sizes[0]]
    n_epochs = [n_epochs[0]]
    num_nodes = [num_nodes[0]]
    hidden_layers = [hidden_layers[0]]



# Define network structure
class LinearNet(nn.Module):
    """
    Defines a feedforward neural network model with linear layers.

    Args:
        input_size (int): Size of the input features.
        num_hid_layers (int): Number of hidden layers.
        layers_size (int): Size of each hidden layer.
        output_size (int): Size of the output.
        activation (torch.nn.Module, optional): Activation function to use. Defaults to nn.ReLU().
        #############################################################################################
        This code is more flexible if you want to easily change activation functions between layers. 
        The original code simplifies the structure but assumes a single type of 
        activation function is used throughout.
    """

    def __init__(self, input_size, num_hid_layers, layers_size, output_size, hl_shape='const', activation=nn.ReLU()):
        super(LinearNet, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Add the first layer
        self.layers.append(nn.Linear(input_size, layers_size))
        self.activations.append(activation)

        if hl_shape == 'const':
            for i in range(1, num_hid_layers):
                self.layers.append(nn.Linear(layers_size, layers_size))
                self.activations.append(activation)
            # Add the output layer without activation
            self.layers.append(nn.Linear(layers_size, output_size))
        elif hl_shape == 'doubling':
            num_hid_layers += 1  # for correct numerics
            for i in range(1, num_hid_layers):
                self.layers.append(nn.Linear(layers_size * i, layers_size * (i + 1)))
                self.activations.append(activation)
            # Add the output layer without activation
            self.layers.append(nn.Linear(layers_size * num_hid_layers, output_size))


    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activations[i](x)
        x = self.layers[-1](x)  # No activation after the final layer
        return x

# Initialize KFold with the number of splits specified in arguments
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

############################################    TRAINING CYCLE    ############################################  

#stores best fold data of best hyper parameter setting
overall_best=BestLossData()

# Create a folder to store training results and get the folder name
foldername = create_training_folder()

# Create a text file to write the used arguments for the training
argsf = os.path.join(foldername, "args.txt")
af = open(argsf, 'w')
print(' '.join(sys.argv), file=af)
af.close()

write_loss_av=True
loss_av=PerParamterLoss()
loss_av.write_header('loss_fn,optimizer,scaler,batch_size,learning_rate,n_epochs,hidden_layers,num_nodes,encoder')


# Get encoded categorical data, mappings, and encoder
cat_data_encoded, ordinal_mappings, onehot_mappings, encoder = get_encoded_cat_data_df(data, cat_feat_names, encoding_method)

# Concatenate numerical and categorical data
all_features = pd.concat([num_data, cat_data_encoded], axis=1)

# Get the list of all feature titles
all_features_titles = all_features.columns.to_list()

# Calculate number of features
num_of_features = len(all_features.columns.to_list())

# Define X,y values
X = all_features.values
y = data[target_name].values

# Create y_fake array in order to track indexes after splitting dataset
y_fake = np.arange(len(y))


#in train mode: maximize training set size
if mode=='train':
    tr_sz=len(X)-1
else:
    tr_sz=0.8

#@TODO: check if the loop can be restructured such that the scaler loop is the most outer loop
#       and if then the itertool multi loop can be used for the remaining hyperparameters
#
# from itertools import product
# ideal:
#
# main loop 
#  scaler layers nodes optimizer learning-rate batch-size
#  for (v3,v2,v1) in product(p3,p2,p1):
#    print(v1,v2,v3)  
#  
# inner loops
#        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
#                for epoch in range(epochs):
#                       for start in batch_starts: 
# Placing the batch_starts tensor outside the fold loop but inside the itertools.product loop would mean 
# that it is generated once per combination of hyperparameters but before the data is split into folds.
# This is problematic because the length of X_fold_train (which determines the size of batch_starts) 
#can vary with each fold due to the k-fold split
# Therefore, generating batch_starts must be done after the k-fold split has determined the training set 
# for each fold.
if True: 

    # Shuffle and split the train and test data indexes
    i_train, i_test, i_train, i_test = train_test_split(y_fake, y_fake, train_size=tr_sz, random_state=42)

    # split the data into training and testing sets
    #X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42) !! bad on random_state=None
    X_train_raw=X[i_train] ; X_test_raw=X[i_test]
    y_train    =y[i_train] ; y_test    =y[i_test]

 


# Make the scaler loop the most outer loop
for scaler in scalers:
    # Check if the scaler is an instance of PowerTransformer
    if isinstance(scaler, PowerTransformer):
        # Initialize a PowerTransformer with method "yeo-johnson" and standardize=False
        scaler = PowerTransformer(method="yeo-johnson", standardize=False)
        # Fit the transformer to the training data
        scaler.fit(X_train_raw)

        # Transform the training data
        X_train = scaler.transform(X_train_raw)
        # Transform the testing data
        X_test = scaler.transform(X_test_raw)

        # Convert the transformed training data to a numpy array
        X_train_np = np.array(X_train)
        # Calculate the mean of the transformed training data along each feature
        X_train_mean = np.mean(X_train_np, axis=0)
        # Calculate the standard deviation of the transformed training data along each feature
        X_train_std = np.std(X_train_np, axis=0)

        # Reinitialize the scaler with standardize=True
        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
        # Fit the transformer to the training data again
        scaler.fit(X_train_raw)

        # Transform the training data again
        X_train = scaler.transform(X_train_raw)
        # Transform the testing data again
        X_test = scaler.transform(X_test_raw)

    # If the scaler is not PowerTransformer
    else:
        # Fit the scaler to the training data
        scaler.fit(X_train_raw)

        # Transform the training data
        X_train = scaler.transform(X_train_raw)
        # Transform the testing data
        X_test = scaler.transform(X_test_raw)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Use itertools.product for the remaining hyperparameters
    for n_midlayers, midlayer_nodes, batch_size, loss_fn, learning_rate, epochs, activation_function in itertools.product(
        hidden_layers, num_nodes, batch_sizes, loss_fn_arr, learning_rates, n_epochs, activation_func
    ):
        
        #activation_name = [name for name, func in activation_functions_dict.items() if func == activation_function][0]

        
        
        # Initialize the model
        model = LinearNet(
            input_size=num_of_features,
            num_hid_layers=n_midlayers,
            layers_size=midlayer_nodes,
            output_size=1,
            hl_shape='doubling',
            activation=activation_function
        )

        # Initialize the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_loss_per_fold = [1000000] * n_folds

        # Iterate over the folds generated by the KFold split
        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            print(f"Fold {fold + 1}")
            print("-------")

            # Split data into training and validation sets for this fold
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

            model.apply(reset_weights)

            # Generate a tensor containing the starting indices of each batch
            batch_starts = torch.arange(0, len(X_fold_train), batch_size)

            start_time = time.time()
            for epoch in range(epochs):

                # Set the model to training mode
                model.train()

                # Iterate over batches
                for start in batch_starts:
                    # Take a batch
                    X_batch = X_fold_train[start:start+batch_size]
                    y_batch = y_fold_train[start:start+batch_size]

                    # Forward pass
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()

                    # Update weights
                    optimizer.step()

                # Evaluate accuracy at end of each epoch
                model.eval()
                y_pred = model(X_fold_val)

                # Calculate validation loss value, take the squared root value if loss function is MSE
                if isinstance(loss_fn, nn.MSELoss):
                    loss_val = math.sqrt(float(loss_fn(y_pred, y_fold_val)))   
                else:
                    loss_val = float(loss_fn(y_pred, y_fold_val))

                # Track the best performing model
                if loss_val < best_loss_per_fold[fold]: 
                    best_loss_per_fold[fold] = loss_val

                if loss_val < overall_best.loss_val:
                    overall_best.loss_val = loss_val
                    overall_best.weights = copy.deepcopy(model.state_dict())
                    overall_best.epoch = epoch
                    overall_best.loss_fn = loss_fn
                    overall_best.optimizer = optimizer
                    overall_best.scaler = scaler
                    overall_best.fold = fold
                    overall_best.batch_size = batch_size
                    overall_best.lr = learning_rate
                    overall_best.n_epochs = epochs
                    overall_best.hidden_layers = n_midlayers
                    overall_best.num_nodes = midlayer_nodes
                    overall_best.encoder = encoder
                    overall_best.fold = fold + 1
                    overall_best.activation = activation_function  #activation_name
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_val}')

        # Write average best loss for one parameter setting
        if write_loss_av:
            loss_av.write_vals(
                np.mean(best_loss_per_fold), 
                loss_fn, optimizer, scaler, batch_size, learning_rate, epochs, n_midlayers, midlayer_nodes, encoder, activation_function #activation_name
            )

        end_time = time.time()

        # Calculate execution time for training process
        exe_time = end_time - start_time


#print(num_of_features,overall_best.hidden_layers,overall_best.num_nodes)
finalmodel=LinearNet(
    input_size=num_of_features,
    num_hid_layers=overall_best.hidden_layers,
    layers_size=overall_best.num_nodes,
    output_size=1,hl_shape='doubling', 
    activation=overall_best.activation #activation_functions_dict[overall_best.activation]

)
finalmodel.load_state_dict(overall_best.weights)

# Switch the model to evaluation mode
finalmodel.eval()
torch.no_grad()

# Get random entry from test set
k = randint(0, len(X_test)-1)

# Provide an example input that would normally be provided to the model's forward() method
example = X_test[k]

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
traced_script_module = torch.jit.trace(finalmodel, example)

# Save the traced model to a file named "best_model.pt"
traced_model_path = os.path.join(foldername, "best_model.pt")
traced_script_module.save(traced_model_path)

# Create file to save pickled best scaler
scaler_path = os.path.join(foldername, "best_scaler.pkl")
# Create file to save pickled best encoder
encoder_path = os.path.join(foldername, "best_encoder.pkl")

# Save pickled scaler
dump(overall_best.scaler, open(scaler_path, 'wb'))
# Save pickled encoder
dump(overall_best.encoder, open(encoder_path, 'wb'))


# Create a text file and write the required preprocessing details of the training for later use
scalerfile = os.path.join(foldername, "preprocessing.txt")
sf = open(scalerfile, 'w')
sf.write("FEATURE ORDER: ")
sf.write(str(all_features_raw.columns.to_list()))
sf.write("\nORDINAL_MAPS: ")
sf.write(str(ordinal_mappings))
sf.write("\nONEHOT_MAPS: ")
sf.write(str(onehot_mappings))
check_scaler_for_preprocessing(overall_best.scaler, sf)


# Create a text file to store the best hyperparameters of the training cycle
besthyperparams = os.path.join(foldername, "best_hyperparameters.txt")
of = open(besthyperparams, 'w')
overall_best.print_vals(of)
of.close()


'''
# Create dataframe containing only the training set
train_df = data.iloc[i_train]

# Create a copy of training validation data df to avoid messing with original df
val_df = train_df.iloc[val_index].copy()

# Add new column to "val_df"
val_df.loc[:, 'train_prediction'] = best_val_pred_values

# Save "val_df" to CSV
val_df.to_csv('test.csv', sep='\t', index=False)
'''

if mode == 'train':
    sys.exit()

###############################################################################################################  

###############################################    TEST MODE    ############################################### 

elif mode == 'test':
    import matplotlib.pyplot as plt
    import shap

    X_test = overall_best.scaler.transform(X_test_raw)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    
    # Set the model to evaluation mode and perform inference without gradient tracking
    finalmodel.eval()
    with torch.no_grad():

        y_actual = []
        y_preds = []

        # Create csv file to save initial data and their predictions
        csv_file = f'preds_output_{batch_sizes}_{learning_rates}_{n_epochs}_{hidden_layers}_{num_nodes}.csv'
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')

            # Define header containing initial column titles plus "y_preds"
            header = initial_feat_file_cols + ['y_preds']
            
            # Write header on file
            writer.writerow(header)

            # Iterate through X_test dataset
            for i in range(len(X_test)):

                # Extract a single raw sample from the test set
                X_sample_raw = X_test_raw[i: i+1]
                # Extract the corresponding scaled test sample
                X_test_scaled = X_test[i: i+1]

                # Perform forward pass to get prediction
                y_pred = finalmodel(X_test_scaled)

                # Store actual and predicted values
                y_actual.append(y_test[i])
                y_preds.append(y_pred.item())

                # Convert raw test sample to a list
                x_sample_list = X_sample_raw[0].tolist()

                # Get the index of the current test sample
                idx = i_test[i]
                # Get the corresponding row from the original data
                row = data.iloc[idx].values.tolist()

                # Append the predicted value to the row
                complete_row = row + [float(y_pred.item())]

                # Write the complete row to the CSV file
                writer.writerow(complete_row)
                    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file, delimiter="\t")

    # Extract the predicted values from the DataFrame
    predicted_values = df['y_preds']
    # Flatten the actual and predicted values lists
    y_actual_flat = np.ravel(y_actual)
    y_preds_flat = np.ravel(y_preds)

    # Calculate the Pearson correlation coefficient
    pearson_corr, _ = pearsonr(y_actual_flat, y_preds_flat)

    # Add the Pearson correlation as text to the plot
    plt.text(max(y_actual_flat), max(y_preds_flat), f'Pearson Correlation: {pearson_corr:.2f}', ha='right', va='top')

    # Create a scatter plot of the actual vs predicted values
    plt.scatter(y_actual, y_preds)
    plt.xlabel('y_actual')
    plt.ylabel('y_preds')
    plt.title('Scatter Plot of y_actual vs y_preds')

    # Fit a trend line to the data
    z = np.polyfit(y_preds_flat, y_actual_flat, 1)
    p = np.poly1d(z)
    # Plot the trend line
    plt.plot(y_preds_flat, p(y_preds_flat), color='black', label=f'Trend line: {z[0]:.2f}x + {z[1]:.2f}')

    # Add a legend to the plot
    plt.legend()
    # Display the plot
    plt.show()

    # Disable JIT compilation for Numba
    os.environ["NUMBA_DISABLE_JIT"] = "1"

   # Wrap the linear regression model with a PyTorch wrapper
    def model_func(input_data):
        # Convert input_data to tensor if necessary
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        # Forward pass through the model
        finalmodel.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = finalmodel(input_tensor)
        
        # Return the output tensor
        return output.numpy()


    # Create DataFrames from numpy arrays
    ###RM X_train_df = pd.DataFrame(X_train_raw)
    X_test_df = pd.DataFrame(X_test)

    # Get feature columns names for SHAP grph
    feature_names = all_features.columns.to_list()
    
    # Define number of samples for SHAP values calculation
    K=50
    bg = shap.sample(X_test_df, K)

    # Initialize the SHAP explainer with the wrX_test_rawapped PyTorch model
    explainer = shap.KernelExplainer(model_func, bg)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(bg)

    # Visualize the SHAP values
    shap.summary_plot(shap_values[0], bg, feature_names)

    # plt.plot(idx, y_preds, label='Predictions')
    # plt.plot(idx, y_actual, label='Actual Values')
    # plt.xlabel('Index (i)')
    # plt.ylabel('ASA Values')
    # plt.title('Predictions vs Actual Values')
    # plt.legend()
    # plt.show()


###############################################################################################################  
