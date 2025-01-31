import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import trange, tqdm
import wandb
import os

# Initialize GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Weights & Biases
wandb.init(project="My_Project", config={
    "epochs": 2000,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "hidden_dim": 128,
    "lstm_hidden_dim": 256,
    "n_hidden_layers": 5,
    "activation_function": "relu",
    "ensemble": 1,
    "seed": 132
})

# Set random seed for reproducibility
np.random.seed(wandb.config["seed"])
torch.manual_seed(wandb.config["seed"])

# Load the input data (combined_data.csv)
input_data = pd.read_csv("combined_data.csv").values  # Shape: (num_samples, 4)

# Load the impedance data (real and imaginary parts)
imp_dd_real = pd.read_csv("remaining_Imp_dd_real.csv")
imp_dd_imag = pd.read_csv("remaining_Imp_dd_imag.csv")

# Normalization function
def normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

# STEP 1: Reshape Input Data
sequence_length = 89  # 89 frequencies per OP set
input_dim = input_data.shape[1] - 1  # Exclude frequency column
num_op_sets = input_data.shape[0] // sequence_length

# Extract OP values and frequency
op_values = input_data[:, :3]
frequencies = input_data[:, 3]

# Normalize OP values and frequencies
op_values, op_mean, op_std = normalize(op_values)
frequencies, freq_mean, freq_std = normalize(frequencies.reshape(-1, 1))

# Reshape OP values and frequency into sequences
op_values_reshaped = op_values.reshape(num_op_sets, sequence_length, -1)
frequencies_reshaped = frequencies.reshape(num_op_sets, sequence_length, 1)

# Combine into a single input tensor and move to GPU
input_tensor = torch.tensor(np.concatenate([op_values_reshaped, frequencies_reshaped], axis=-1), dtype=torch.float32).to(device)

# STEP 2: Reshape Output Data
imp_dd_real_values = imp_dd_real.iloc[:, 1:].values
imp_dd_imag_values = imp_dd_imag.iloc[:, 1:].values

# Normalize impedance values
imp_dd_real_values, real_mean, real_std = normalize(imp_dd_real_values)
imp_dd_imag_values, imag_mean, imag_std = normalize(imp_dd_imag_values)

# Transpose and reshape impedance values
real_imp_reshaped = imp_dd_real_values.T.reshape(num_op_sets, sequence_length, 1)
imag_imp_reshaped = imp_dd_imag_values.T.reshape(num_op_sets, sequence_length, 1)

# Combine into a single output tensor and move to GPU
output_tensor = torch.tensor(np.concatenate([real_imp_reshaped, imag_imp_reshaped], axis=-1), dtype=torch.float32).to(device)

# STEP 3: Set Up DataLoader
train_size = int(0.8 * len(input_tensor))
test_size = len(input_tensor) - train_size
train_dataset, test_dataset = random_split(TensorDataset(input_tensor, output_tensor), [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=wandb.config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=wandb.config["batch_size"], shuffle=False)

print("DataLoader setup completed.")

class MLPWithLSTM(nn.Module):
    def __init__(self, input_dim=4, output_dim=2, hidden_dim=wandb.config["hidden_dim"], lstm_hidden_dim=wandb.config["lstm_hidden_dim"], n_hidden_layers=wandb.config["n_hidden_layers"], use_dropout=False, activation_function=wandb.config["activation_function"]):
        super().__init__()
        self.use_dropout = use_dropout
        self.activation_function = activation_function.lower()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, batch_first=True)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1)

        self.layer_sizes = [lstm_hidden_dim] + n_hidden_layers * [hidden_dim] + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)])

    def forward(self, input_sequence):
        lstm_out, _ = self.lstm(input_sequence)
        hidden = lstm_out
        for layer in self.layers[:-1]:
            if self.activation_function == 'relu':
                hidden = torch.relu(layer(hidden))
            if self.use_dropout:
                hidden = self.dropout(hidden)
        return self.layers[-1](hidden)

# Define training function
def train_and_evaluate(net, train_loader, test_loader, epochs=wandb.config["epochs"]):
    net.to(device)  # Move model to GPU
    optimizer = torch.optim.Adam(net.parameters(), lr=wandb.config["learning_rate"])
    criterion = nn.MSELoss()
    wandb.watch(net, log="all")

    train_losses, test_losses = [], []
    progress_bar = trange(epochs)

    for epoch in progress_bar:
        net.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move batch to GPU
            optimizer.zero_grad()
            predictions = net(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch})

        # Evaluate on test data
        net.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move batch to GPU
                predictions = net(x_batch)
                loss = criterion(predictions, y_batch)
                epoch_test_loss += loss.item()

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        wandb.log({"test_loss": avg_test_loss, "epoch": epoch})

        progress_bar.set_postfix(train_loss=f'{avg_train_loss:.4f}', test_loss=f'{avg_test_loss:.4f}')

    return train_losses, test_losses

# Train ensemble models
ensemble_size = wandb.config["ensemble"]
ensemble = [MLPWithLSTM(input_dim=input_tensor.shape[2], output_dim=output_tensor.shape[2]).to(device) for _ in range(ensemble_size)]

all_train_losses = []
all_test_losses = []
all_ensemble_preds = []

for idx, net in enumerate(ensemble):
    print(f"Training model {idx + 1}...")
    train_losses, test_losses = train_and_evaluate(net, train_loader, test_loader)
    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)

    # Generate and store predictions for the ensemble
    net.eval()
    predictions = net(input_tensor).detach().cpu().numpy()
    all_ensemble_preds.append(predictions)

# Save and log results
plt.figure(figsize=(10, 6))
plt.plot(np.mean(all_train_losses, axis=0), label='Mean Train Loss')
plt.plot(np.mean(all_test_losses, axis=0), label='Mean Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig('train_test_loss.png')
wandb.log({'train_test_loss.png': wandb.Image('train_test_loss.png')})



# Save ensemble predictions to CSV
def save_ensemble_predictions_to_csv(frequencies, ensemble_preds, output_file):
    """
    Optimized function to save ensemble predictions to a CSV file.
    """
    import time
    start_time = time.time()

    frequencies = np.ravel(frequencies)  # Ensure 1D array
    ensemble_preds_np = np.array(ensemble_preds)  # Convert to NumPy array

    # Dynamically determine `num_ops` and `seq_length`
    ensemble_size, num_ops, seq_length, _ = ensemble_preds_np.shape  # Shape: (ensemble_size, num_ops, 89, 2)

    print(f"Detected {num_ops} operating points, {seq_length} frequency steps.")

    # Compute mean and standard deviation efficiently
    mean_real_preds = np.mean(ensemble_preds_np[:, :, :, 0], axis=0)  # Shape: (num_ops, 89)
    std_real_preds = np.std(ensemble_preds_np[:, :, :, 0], axis=0)    # Shape: (num_ops, 89)
    mean_imag_preds = np.mean(ensemble_preds_np[:, :, :, 1], axis=0)  # Shape: (num_ops, 89)
    std_imag_preds = np.std(ensemble_preds_np[:, :, :, 1], axis=0)    # Shape: (num_ops, 89)

    # **Ensure all arrays are 1D before adding to DataFrame**
    mean_real_preds = mean_real_preds.flatten()
    std_real_preds = std_real_preds.flatten()
    mean_imag_preds = mean_imag_preds.flatten()
    std_imag_preds = std_imag_preds.flatten()

    # **Fix: Ensure `frequencies_subset` has correct length**
    frequencies_subset = np.tile(frequencies[:seq_length], num_ops).flatten()

    # **Debugging Print Statements**
    print(f"Expected DataFrame Length: {num_ops * seq_length}")
    print(f"Length of frequencies_subset: {len(frequencies_subset)}")
    print(f"Shape of mean_real_preds: {mean_real_preds.shape} (Expected: {num_ops * seq_length})")
    print(f"Shape of std_real_preds: {std_real_preds.shape} (Expected: {num_ops * seq_length})")
    print(f"Shape of mean_imag_preds: {mean_imag_preds.shape} (Expected: {num_ops * seq_length})")
    print(f"Shape of std_imag_preds: {std_imag_preds.shape} (Expected: {num_ops * seq_length})")

    # **Ensure all arrays have the same length**
    expected_length = num_ops * seq_length
    assert len(frequencies_subset) == expected_length, f"Expected {expected_length} frequencies, got {len(frequencies_subset)}"
    assert mean_real_preds.shape == (expected_length,), f"mean_real_preds has shape {mean_real_preds.shape}, expected ({expected_length},)"
    assert std_real_preds.shape == (expected_length,), f"std_real_preds has shape {std_real_preds.shape}, expected ({expected_length},)"
    assert mean_imag_preds.shape == (expected_length,), f"mean_imag_preds has shape {mean_imag_preds.shape}, expected ({expected_length},)"
    assert std_imag_preds.shape == (expected_length,), f"std_imag_preds has shape {std_imag_preds.shape}, expected ({expected_length},)"

    # **Build dictionary for DataFrame**
    data_dict = {
        "Frequency": frequencies_subset,
        "Mean_Real_Prediction": mean_real_preds,
        "Std_Real_Prediction": std_real_preds,
        "Mean_Imag_Prediction": mean_imag_preds,
        "Std_Imag_Prediction": std_imag_preds
    }

    # **Convert to DataFrame once**
    pred_df = pd.DataFrame(data_dict)

    # **Save as CSV**
    pred_df.to_csv(output_file, index=False)

    end_time = time.time()
    print(f"CSV saved in {end_time - start_time:.2f} seconds.")



# Example usage:
save_ensemble_predictions_to_csv(frequencies, all_ensemble_preds, 'optimized_ensemble_predictions.csv')


# Plot ensemble predictions for real or imaginary impedance
def plot_ensemble_predictions(frequencies, ensemble_preds, true_data, op_index, file_name, impedance_type='real'):
    """
    Plots the ensemble model predictions for a specific operating point (OP).
    """
    plt.figure(figsize=(10, 6))

    # Extract the correct subset of frequencies (only 89 per OP)
    frequencies_subset = frequencies[op_index * 89:(op_index + 1) * 89]

    for i, pred in enumerate(ensemble_preds):
        pred_for_op = pred[op_index, :, 0] if impedance_type == 'real' else pred[op_index, :, 1]

        # Ensure dimensions match before plotting
        if pred_for_op.shape[0] != frequencies_subset.shape[0]:
            print(f"Warning: Mismatched dimensions: frequencies ({frequencies_subset.shape[0]}) vs. pred_for_op ({pred_for_op.shape[0]})")
            continue  # Skip plotting if there's a mismatch

        plt.plot(frequencies_subset, pred_for_op, label=f'Ensemble Model {i+1}', linestyle='--', alpha=0.6)

    plt.plot(frequencies_subset, true_data, 'k-', linewidth=3, label=f"True Function ({impedance_type.capitalize()} Impedance)")
    plt.xlabel('Frequency')
    plt.ylabel(f'{impedance_type.capitalize()} Impedance')
    plt.title(f'Ensemble Predictions vs. True Function for OP {op_index} ({impedance_type.capitalize()})')
    plt.legend(loc="upper right")
    plt.savefig(file_name)
    wandb.log({file_name: wandb.Image(file_name)})


# Example usage: Plot ensemble predictions for real impedance
op_index = 0
true_data_real = imp_dd_real_values[:, op_index]  # Get true data for the specified OP (Real)
plot_ensemble_predictions(frequencies, all_ensemble_preds, true_data_real, op_index, 'ensemble_predictions_op0_real.png', 'real')

# Example usage: Plot ensemble predictions for imaginary impedance
true_data_imag = imp_dd_imag_values[:, op_index]  # Get true data for the specified OP (Imaginary)
plot_ensemble_predictions(frequencies, all_ensemble_preds, true_data_imag, op_index, 'ensemble_predictions_op0_imag.png', 'imag')

# Plot ensemble predictions with uncertainty bands (real or imaginary impedance)
def plot_ensemble_uncertainty(frequencies, ensemble_preds, true_data, op_index, file_name, impedance_type='real'):
    """
    Plots ensemble predictions with uncertainty bands for a specific operating point (OP).
    """
    plt.figure(figsize=(10, 6))

    # Extract the correct subset of frequencies (only 89 per OP)
    frequencies_subset = np.ravel(frequencies)[op_index * 89:(op_index + 1) * 89]

    # Convert ensemble_preds to numpy array
    ensemble_preds_np = np.array(ensemble_preds)
    
    if impedance_type == 'real':
        preds_for_op = ensemble_preds_np[:, op_index, :, 0]  # Real part
    else:
        preds_for_op = ensemble_preds_np[:, op_index, :, 1]  # Imaginary part

    # Compute mean and std over ensemble models
    y_mean = preds_for_op.mean(axis=0)
    y_std = preds_for_op.std(axis=0)

    # Ensure dimensions match before plotting
    if y_mean.shape[0] != frequencies_subset.shape[0]:
        print(f"Warning: Mismatched dimensions: frequencies ({frequencies_subset.shape[0]}) vs. y_mean ({y_mean.shape[0]})")
        return

    plt.plot(frequencies_subset, true_data, 'k-', linewidth=3, label=f"True Function ({impedance_type.capitalize()} Impedance)")
    plt.plot(frequencies_subset, y_mean, label="Predictive Mean", color="#408765", linewidth=3, linestyle="dashed")
    plt.fill_between(frequencies_subset, y_mean - 2 * y_std, y_mean + 2 * y_std, alpha=0.3, color="#86cfac")
    
    plt.xlabel('Frequency')
    plt.ylabel(f'{impedance_type.capitalize()} Impedance')
    plt.title(f'Ensemble Predictions with Uncertainty Bands for OP {op_index} ({impedance_type.capitalize()})')
    plt.legend(loc="upper right")
    plt.savefig(file_name)
    wandb.log({file_name: wandb.Image(file_name)})


# Example usage: Plot ensemble uncertainty for real impedance
plot_ensemble_uncertainty(frequencies, all_ensemble_preds, true_data_real, op_index, 'ensemble_uncertainty_real.png', 'real')

# Example usage: Plot ensemble uncertainty for imaginary impedance
plot_ensemble_uncertainty(frequencies, all_ensemble_preds, true_data_imag, op_index, 'ensemble_uncertainty_imag.png', 'imag')

# Function to predict impedance for new operational points across all frequencies
def predict_impedance_for_new_op(new_op, frequencies, ensemble_models, mean_train=None, std_train=None, mean_freq=None, std_freq=None):
   
    # Normalize new_op using the training data's mean and std (excluding frequency)
    if mean_train is not None and std_train is not None:
        new_op = (np.array(new_op) - mean_train) / std_train  # Normalize operational point variables
    
    ensemble_predictions_real = []
    ensemble_predictions_imag = []

    for frequency in frequencies:
        # Normalize frequency separately using its own mean and std
        normalized_frequency = (frequency - mean_freq) / std_freq if mean_freq is not None else frequency
        new_input = torch.tensor(new_op.tolist() + [normalized_frequency], dtype=torch.float32).unsqueeze(0).to(device)

        predictions_for_freq = []

        # Get predictions from each model in the ensemble
        for model in ensemble_models:
            model.eval()
            with torch.no_grad():
                prediction = model(new_input).cpu().numpy()
                predictions_for_freq.append(prediction)

        # Convert to NumPy array for easier manipulation
        predictions_for_freq = np.array(predictions_for_freq)

        # Calculate mean and standard deviation for the current frequency
        mean_prediction = predictions_for_freq.mean(axis=0).squeeze()

        # Append to the lists (real and imaginary parts)
        ensemble_predictions_real.append(mean_prediction[0])
        ensemble_predictions_imag.append(mean_prediction[1])

    return np.array(ensemble_predictions_real), np.array(ensemble_predictions_imag)

# Function to plot predictions for real and imaginary impedance along with true values
def plot_new_op_predictions(frequencies, predictions, true_values, file_name, impedance_type='real'):
    plt.figure(figsize=(10, 6))
    
    frequencies, freq_mean, freq_std = normalize(frequencies.reshape(-1, 1))
    # Plot predictive mean
    plt.plot(frequencies, predictions, label="Predictive Mean", color="#408765", linewidth=3)

    # Plot true values
    normalized_true_data, true_data_mean, true_data_std = normalize(true_values)
    plt.plot(frequencies, normalized_true_data, label="True Value", color="red", linestyle="--", linewidth=2)
    
    plt.xlabel('Frequency')
    plt.ylabel(f'{impedance_type.capitalize()} Impedance')
    plt.title(f'Predictions vs True Values for New OP ({impedance_type.capitalize()})')
    plt.legend(loc="upper right")
    plt.savefig(file_name)
    wandb.log({file_name: wandb.Image(file_name)})

# Load true impedance values from CSV files (assuming same format as before)
def load_true_impedance(csv_file):
    data = pd.read_csv(csv_file)
    frequencies = data.iloc[:, 0].values  # First column is frequency
    true_values = data.iloc[:, 1].values  # Second column is the true impedance value
    return frequencies, true_values

# Example usage:
# Load true impedance values for the real and imaginary parts
true_frequencies_real, true_values_real = load_true_impedance('extracted_Imp_dd_real.csv')  # Load true real impedance
true_frequencies_imag, true_values_imag = load_true_impedance('extracted_Imp_dd_imag.csv')  # Load true imaginary impedance

# Example operational points (new input for prediction)
new_op = [117.7, -211.86, -58.85]  # Example operational points

# Assuming `input_data` is a NumPy array with the last column as the frequency
# Example: input_data = np.load("your_training_data.npy")

# Compute mean and standard deviation of training data for normalization
normalized_data, mean_train, std_train = normalize(input_data[:, :-1])  # Exclude frequency for normalization
mean_freq = input_data[:, -1].mean()  # Mean of frequency column
std_freq = input_data[:, -1].std()  # Std of frequency column

# Get predictions for the new operational points
mean_preds_real, mean_preds_imag = predict_impedance_for_new_op(new_op, true_frequencies_real, ensemble, mean_train, std_train, mean_freq, std_freq)

# Plot the predictions for real impedance along with true values
plot_new_op_predictions(true_frequencies_real, mean_preds_real, true_values_real, file_name='New_OP_prediction_real.png', impedance_type='real')

# Plot the predictions for imaginary impedance along with true values
plot_new_op_predictions(true_frequencies_imag, mean_preds_imag, true_values_imag, file_name='New_OP_prediction_imag.png', impedance_type='imag')
