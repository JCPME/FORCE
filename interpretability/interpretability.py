# %%
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import random
import warnings
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from captum.attr import Saliency, IntegratedGradients
from tqdm import tqdm
from typing import Tuple, List, Dict, Any
import numpy as np

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def stft_features_from_snippet(
    translation_array,   # shape (N, 3)
    rotation_array,      # shape (N, 4)
    timestamp_array,     # shape (N,)
    n_fft=256,
    hop_length=128,
    fixed_time_frames=1500  # Padding to 1500 time frames
):
    """
    Returns a 2D time-frequency representation for each of the 7 axes,
    then merges them so TCN can process (channels, time_frames).

    Parameters:
        translation_array (np.ndarray): Shape (N, 3)
        rotation_array (np.ndarray): Shape (N, 4)
        timestamp_array (np.ndarray): Shape (N,)
        n_fft (int): Number of FFT components
        hop_length (int): Number of samples between successive frames
        fixed_time_frames (int): Desired number of time frames after padding/truncation

    Returns:
        np.ndarray: 2D array of shape [(7 * freq_bins), fixed_time_frames]
    """
    import torch
    import numpy as np

    n = len(translation_array)
    if n < 2:
        # Return an empty placeholder if there's too little data
        return np.zeros((7 * (n_fft // 2 + 1), fixed_time_frames), dtype=np.float32)

    # Convert to np arrays if they aren't already
    translation_array = np.array(translation_array)  # (n, 3)
    rotation_array = np.array(rotation_array)        # (n, 4)

    # 1) Interpolate the snippet to uniform sampling
    t0, t_end = timestamp_array[0], timestamp_array[-1]
    new_t = np.linspace(t0, t_end, n)

    trans = np.zeros((n, 3))
    rot   = np.zeros((n, 4))

    for i in range(3):
        trans[:, i] = np.interp(new_t, timestamp_array, translation_array[:, i])
    for i in range(4):
        rot[:, i]   = np.interp(new_t, timestamp_array, rotation_array[:, i])

    # 2) Convert to torch tensors
    trans_torch = torch.tensor(trans, dtype=torch.float32)  # shape: (n, 3)
    rot_torch   = torch.tensor(rot,   dtype=torch.float32)  # shape: (n, 4)

    # 3) Define a helper to compute STFT magnitude with Hann window
    def stft_mag(signal_1d):
        # Zero-pad if the signal is shorter than n_fft
        if signal_1d.size(0) < n_fft:
            pad_size = n_fft - signal_1d.size(0)
            signal_1d = F.pad(signal_1d, (0, pad_size), 'constant', 0)
        # shape: [n]
        # returns shape: [freq_bins, time_frames]
        window = torch.hann_window(n_fft)
        stft_res = torch.stft(
            signal_1d,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            return_complex=False
        )
        # stft_res => [freq_bins, time_frames, 2]
        real_part = stft_res[..., 0]
        imag_part = stft_res[..., 1]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        return magnitude  # shape: (freq_bins, time_frames)

    # 4) Compute STFT for each axis => 7 total
    all_mags = []

    # translation => x,y,z
    for i in range(3):
        mag = stft_mag(trans_torch[:, i])  # shape: (freq_bins, time_frames)
        all_mags.append(mag.unsqueeze(0))

    # rotation => w,x,y,z
    for i in range(4):
        mag = stft_mag(rot_torch[:, i])    # shape: (freq_bins, time_frames)
        all_mags.append(mag.unsqueeze(0))

    # 5) Concatenate along channel dimension
    cat_mags = torch.cat(all_mags, dim=0)  # shape: [7, freq_bins, time_frames]

    # 6) Reshape to [7 * freq_bins, time_frames]
    stft_2d = cat_mags.view(7 * (n_fft//2 + 1), -1).numpy()  # shape: [7*(freq_bins), time_frames]

    # 7) Pad or truncate to fixed_time_frames
    current_time_frames = stft_2d.shape[1]
    if current_time_frames < fixed_time_frames:
        # Pad with zeros at the end
        padding = fixed_time_frames - current_time_frames
        stft_2d = np.pad(stft_2d, ((0, 0), (0, padding)), mode='constant')
    elif current_time_frames > fixed_time_frames:
        # Truncate to fixed_time_frames
        stft_2d = stft_2d[:, :fixed_time_frames]

    return stft_2d.astype(np.float32)

# ==================================
# Data Loading
# ==================================
data_path = '/Users/julien/Library/Mobile Documents/com~apple~CloudDocs/ETH/Master-Season-1/Deep Learning/project/output_with_bones.pkl'
df = pd.read_pickle(data_path)

# Check DataFrame columns and sample data
print("Columns in DataFrame:", df.columns)


# ------------------------------
# 2) Create Binary Label (above/below mean avg_grs_score)
# ------------------------------
mean_score = df['avg_grs_score'].mean()
df['label'] = (df['avg_grs_score'] >= mean_score).astype(int)

# Get unique participants
unique_participants = df['participant_num'].unique()

unique_participants = unique_participants.tolist()
random.shuffle(unique_participants)

# Filter the DataFrame for the specified tools
tools_to_include = ['Ulna_1', 'Ulna_2', 'Radius_1', 'Radius_2']
filtered_df = df[df['tool'].isin(tools_to_include)]
print("\nFiltered DataFrame Description:")
print(filtered_df.describe())

# ==================================
# Feature Extraction and Scaling
# ==================================
n_fft = 256
hop_length = 128

def build_feature_matrix(input_df,
                        tool_encoder=None,
                        case_encoder=None,
                        n_fft=256,
                        hop_length=128):
    if tool_encoder is None:
        tool_encoder = LabelEncoder().fit(input_df['tool'].astype(str))
    if case_encoder is None:
        case_encoder = LabelEncoder().fit(input_df['case'].astype(str))

    X_list = []
    y_list = []

    for idx, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Building Features"):
        # 1) Get STFT for this snippet
        stft_2d = stft_features_from_snippet(
            row['translation_array'],
            row['rotation_array'],
            row['timestamp_array'],
            n_fft=n_fft,
            hop_length=hop_length,
            fixed_time_frames=1500  # Padding to 1500
        )
        # shape: [channels, time_frames]

        # 2) Additional numeric features
        tool_val  = tool_encoder.transform([str(row['tool'])])[0]
        case_val  = case_encoder.transform([str(row['case'])])[0]
        tpt_val   = row['total_procedure_time']

        # We'll store these 3 as a separate vector
        extra_feats = np.array([tool_val, case_val, tpt_val], dtype=np.float32)

        label = row['label']

        # 3) Save
        # We'll store stft_2d as is; we won't flatten it. We can handle flattening or not in the dataset later.
        X_list.append((stft_2d, extra_feats))
        y_list.append(label)

    return X_list, np.array(y_list).astype(int), tool_encoder, case_encoder

def build_feature_matrix_test(input_df,
                              tool_encoder,
                              case_encoder,
                              scaler_stft,
                              scaler_extra,
                              n_fft=256,
                              hop_length=128):
    """
    Constructs feature matrices using STFT features and additional numeric features.
    Assumes encoders and scalers are already fitted.
    """
    X_list = []
    y_list = []

    for idx, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Building Test Features"):
        stft_2d = stft_features_from_snippet(
            row['translation_array'],
            row['rotation_array'],
            row['timestamp_array'],
            n_fft=n_fft,
            hop_length=hop_length,
            fixed_time_frames=1500  # Padding to 1500
        )
        # shape: [channels, time_frames]

        # Additional numeric features
        tool_val = tool_encoder.transform([str(row['tool'])])[0]
        case_val = case_encoder.transform([str(row['case'])])[0]
        tpt_val  = row['total_procedure_time']

        extra_feats = np.array([tool_val, case_val, tpt_val], dtype=np.float32)

        label = row['label']

        # Scale stft_2d
        stft_2d_flat = stft_2d.flatten().reshape(1, -1)
        stft_2d_scaled_flat = scaler_stft.transform(stft_2d_flat)
        stft_2d_scaled = stft_2d_scaled_flat.reshape(stft_2d.shape)

        # Scale extra_feats
        extra_feats_scaled = scaler_extra.transform(extra_feats.reshape(1, -1)).flatten()

        # Append to list as a tuple
        X_list.append((stft_2d_scaled, extra_feats_scaled))
        y_list.append(label)

    return X_list, np.array(y_list).astype(int)

# Build feature matrices for training
X_analysis, y_analysis, tool_encoder, case_encoder = build_feature_matrix(
    filtered_df,
    tool_encoder=None,
    case_encoder=None,
    n_fft=n_fft,
    hop_length=hop_length
)

print("\nNumber of Augmented Train Samples:", len(X_analysis))
print("Number of Augmented Train Labels:", y_analysis.shape)

# Extract stft_2d and extra_feats from X_train
stft_train = np.array([x[0] for x in X_analysis])  # Shape: (num_samples, channels, time_frames)
extra_train = np.array([x[1] for x in X_analysis]) # Shape: (num_samples, 3)

# Initialize scalers
scaler_stft = StandardScaler()
scaler_extra = StandardScaler()

# Fit scalers on training data
num_samples, channels, time_frames = stft_train.shape
stft_train_reshaped = stft_train.reshape(num_samples, -1)
with tqdm(total=2, desc="Fitting Scalers") as pbar:
    scaler_stft.fit(stft_train_reshaped)
    pbar.update(1)
    scaler_extra.fit(extra_train)
    pbar.update(1)

extra_train_scaled = None
with tqdm(total=2, desc="Scaling Data") as pbar:
    stft_train_scaled = scaler_stft.transform(stft_train_reshaped).reshape(num_samples, channels, time_frames)
    pbar.update(1)
    extra_train_scaled = scaler_extra.transform(extra_train)
    pbar.update(1)

# Recreate X_train as list of tuples with scaled features
X_analysis = list(zip(stft_train_scaled, extra_train_scaled))

# ==================================
# Define Dataset Class
# ==================================
class FFTAdditionalDataset(Dataset):
    def __init__(self, X, y):
        """
        X is now a list of (stft_2d, extra_feats)
        stft_2d: shape [channels, time_frames]
        extra_feats: shape [3,]
        y: labels
        """
        self.X = X   # list of tuples
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        stft_2d, extra_feats = self.X[idx]
        # Convert stft_2d to torch
        stft_2d = torch.tensor(stft_2d, dtype=torch.float32)
        # shape => [channels, time_frames]

        # Convert extra_feats to torch
        extra_feats = torch.tensor(extra_feats, dtype=torch.float32)

        label = self.y[idx]
        return stft_2d, extra_feats, label

# %%
# ==================================
# Create Datasets and DataLoaders
# ==================================
# Create Datasets
train_dataset = FFTAdditionalDataset(X_analysis, y_analysis)


# Define DataLoaders
batch_size = 32

analysis_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


print("\nDataLoaders Created:")
print(f"Train Loader: {len(analysis_loader)} batches")


class Chomp1d(nn.Module):
    """
    Removes the last elements to ensure causality.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2,
                 use_attention=True, n_heads=1):
        super(TemporalBlock, self).__init__()
        self.use_attention = use_attention
        self.n_heads = n_heads

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # If in_channels != out_channels, we use a 1×1 conv to match dimensions
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

        self.init_weights()

        # >>> ATTENTION LAYER <<<
        if self.use_attention:
            self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=self.n_heads, batch_first=False)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x, return_attention=False):
        """
        x => [batch_size, in_channels, seq_len]
        out => [batch_size, out_channels, seq_len]
        """
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)  # changed 'x' to 'out'
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # out shape => [batch_size, out_channels, seq_len]

        attn_weights = None
        # >>> SELF-ATTENTION PER BLOCK <<<
        if self.use_attention:
            # multihead attention wants shape [seq_len, batch_size, embed_dim]
            out_for_attn = out.permute(2, 0, 1)    # => [seq_len, batch, out_channels]
            attn_out, attn_weights = self.attn(out_for_attn, out_for_attn, out_for_attn)
            # shape of attn_out => [seq_len, batch, out_channels]
            # let's add a residual connection from 'out_for_attn'
            out_for_attn = out_for_attn + attn_out
            # back to [batch, out_channels, seq_len]
            out = out_for_attn.permute(1, 2, 0)

        # Residual (skip connection)
        res = x if self.downsample is None else self.downsample(x)

        if return_attention:
            return self.relu(out + res), attn_weights
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2, use_attention=True, n_heads=1):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        in_channels = input_size
        for i in range(num_levels):
            out_channels = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                    use_attention=use_attention,
                    n_heads=n_heads
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_list = []
            out, attn_weights = self.network[0](x, return_attention=True)
            attn_list.append(attn_weights)
            for layer in self.network[1:]:
                out, attn_weights = layer(out, return_attention=True)
                attn_list.append(attn_weights)
            return out, attn_list
        return self.network(x)

class TCN_FFN_Model(nn.Module):
    def __init__(self, tcn_input_channels, num_channels, kernel_size=3, dropout=0.2,
                 additional_input_size=3, ffn_hidden_sizes=[32,16],
                 use_attention=True, n_heads=1):
        super(TCN_FFN_Model, self).__init__()
        self.tcn = TCN(
            input_size=tcn_input_channels,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_attention=use_attention,
            n_heads=n_heads
        )
        self.tcn_output_size = num_channels[-1]

        # FFN for extra features
        self.ffn = nn.Sequential(
            nn.Linear(additional_input_size, ffn_hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ffn_hidden_sizes[0], ffn_hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.ffn_output_size = ffn_hidden_sizes[-1]

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.tcn_output_size + self.ffn_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, stft_features, additional_features, return_attention=False):
        """
        stft_features: [batch_size, channels, time_frames]
        additional_features: [batch_size, 3]
        """
        if return_attention:
            tcn_out, attn_list = self.tcn(stft_features, return_attention=True)  # tcn_out: [batch, out_channels, seq_len]
        else:
            tcn_out = self.tcn(stft_features)  # [batch, out_channels, seq_len]

        tcn_out = torch.mean(tcn_out, dim=2)  # [batch_size, out_channels]

        ffn_out = self.ffn(additional_features)  # [batch_size, ffn_output_size]

        combined = torch.cat((tcn_out, ffn_out), dim=1)  # [batch_size, out_channels + ffn_output_size]
        out = self.classifier(combined)  # [batch_size, 1]

        if return_attention:
            return out, attn_list
        return out

# ==================================
# Load the Trained Model
# ==================================
# Initialize your model architecture
model = TCN_FFN_Model(
    tcn_input_channels=903,  # 7 * 129
    num_channels=[64, 64, 64],
    kernel_size=3,
    dropout=0.2,
    additional_input_size=3,
    ffn_hidden_sizes=[64, 32],
    use_attention=True,
    n_heads=1
)

class ModelWrapperExtra(torch.nn.Module):
    """
    Wrapper for attributing to additional_feats.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, additional_feats, stft_features):
        # Note the order: additional_feats is now the main input
        return self.model(stft_features, additional_feats)


class ModelWrapperStft(torch.nn.Module):
    """
    Wrapper for attributing to stft_features.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, stft_features, additional_feats):
        return self.model(stft_features, additional_feats)

# Load the trained weights
model.load_state_dict(torch.load("/Users/julien/Library/Mobile Documents/com~apple~CloudDocs/ETH/Master-Season-1/Deep Learning/project/best_tcn_ffn_model (1).pth", map_location=device))
model.to(device)
model.eval()

# Wrap the model for Captum
model_wrapper_stft = ModelWrapperStft(model).to(device)
model_wrapper_extra = ModelWrapperExtra(model).to(device)

# ==================================
# Define Attribution Extraction Functions
# ==================================

def extract_feature_importance_stft(
    model_wrapper: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    method: str = 'saliency'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature attributions for STFT features using Captum, store corresponding labels, inputs, and procedure times.

    Parameters:
        model_wrapper (torch.nn.Module): The wrapped model for Captum.
        dataloader (DataLoader): DataLoader to iterate over.
        device (torch.device): Device to perform computations on.
        method (str): Attribution method ('saliency' or 'integrated_gradients').

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Attributions for STFT features with shape [N, 903, time_frames].
            - Corresponding labels with shape [N].
            - STFT inputs with shape [N, 903, time_frames].
            - Procedure times with shape [N].
    """
    if method == 'saliency':
        attr_method = Saliency(model_wrapper)
    elif method == 'integrated_gradients':
        attr_method = IntegratedGradients(model_wrapper)
    else:
        raise ValueError("Unsupported method. Choose 'saliency' or 'integrated_gradients'.")

    all_attributions_stft = []
    all_labels = []
    all_stfts = []
    all_procedure_times = []

    model_wrapper.eval()

    for batch in tqdm(dataloader, desc=f"Extracting {method} attributions with procedure times"):
        stft_features, additional_feats, labels = batch

        # Extract procedure times
        procedure_times = additional_feats[:, 2].detach().cpu().numpy()  # Assuming procedure time is the 3rd feature

        # Move data to the device
        stft_features = stft_features.to(device)
        additional_feats = additional_feats.to(device)
        labels = labels.to(device)

        # Enable gradient computation if using gradient-based methods
        if method in ['saliency', 'integrated_gradients']:
            stft_features.requires_grad = True
            additional_feats.requires_grad = False

        # Compute attributions
        attributions = attr_method.attribute(
            stft_features,
            target=0,
            additional_forward_args=(additional_feats,)
        )

        # Move attributions, inputs, and labels to CPU and convert to NumPy
        all_attributions_stft.append(attributions.detach().cpu().numpy())
        all_stfts.append(stft_features.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_procedure_times.append(procedure_times)

    # Concatenate all arrays
    all_attributions_stft = np.concatenate(all_attributions_stft, axis=0)
    all_stfts = np.concatenate(all_stfts, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_procedure_times = np.concatenate(all_procedure_times, axis=0)

    return all_attributions_stft, all_labels, all_stfts, all_procedure_times


def extract_feature_importance_extra(
    model_wrapper_extra: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    method: str = 'saliency'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature attributions for additional features using Captum and store corresponding labels.

    Parameters:
        model_wrapper_extra (torch.nn.Module): The wrapped model for Captum.
        dataloader (DataLoader): DataLoader to iterate over.
        device (torch.device): Device to perform computations on.
        method (str): Attribution method ('saliency' or 'integrated_gradients').

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Attributions for additional features with shape [N, 3].
            - Corresponding labels with shape [N].
    """
    if method == 'saliency':
        attr_method = Saliency(model_wrapper_extra)
    elif method == 'integrated_gradients':
        attr_method = IntegratedGradients(model_wrapper_extra)
    else:
        raise ValueError("Unsupported method. Choose 'saliency' or 'integrated_gradients'.")

    all_attributions_extra = []
    all_labels = []

    model_wrapper_extra.eval()

    for batch in tqdm(dataloader, desc=f"Extracting {method} attributions for Additional Feats"):
        stft_features, additional_feats, labels = batch

        # Move data to the device
        stft_features = stft_features.to(device)
        additional_feats = additional_feats.to(device)
        labels = labels.to(device)  # Ensure labels are on the correct device

        # Enable gradient computation if using gradient-based methods
        if method in ['saliency', 'integrated_gradients']:
            stft_features.requires_grad = False  # Not attributing to stft_features here
            additional_feats.requires_grad = True

        # Compute attributions
        attributions = attr_method.attribute(
            additional_feats,  # Main input
            target=0,         # Target class (fixed for single-output sigmoid)
            additional_forward_args=(stft_features,)  # Additional inputs
        )

        # Move attributions to CPU and convert to NumPy
        attrib_extra_batch = attributions.detach().cpu().numpy()

        # Append attributions and labels to the lists
        all_attributions_extra.append(attrib_extra_batch)
        all_labels.append(labels.detach().cpu().numpy())

    # Concatenate all attributions and labels
    all_attributions_extra = np.concatenate(all_attributions_extra, axis=0)  # Shape: [N, 3]
    all_labels = np.concatenate(all_labels, axis=0)                          # Shape: [N]

    return all_attributions_extra, all_labels


def extract_attention_weights(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> List[dict]:
    """
    Extract attention weights from all layers for each sample in the dataloader and store labels.

    Parameters:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader to iterate over.
        device (torch.device): Device to perform computations on.

    Returns:
        List[dict]: Each dictionary contains 'label', 'pred', 'prob', and 'attentions' for a sample.
    """
    model.eval()
    attention_data = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Attention Weights"):
            stft_features, additional_feats, labels = batch

            # Move data to device
            stft_features = stft_features.to(device)
            additional_feats = additional_feats.to(device)
            labels = labels.to(device)

            # Forward pass with attention
            outputs, attn_list = model(stft_features, additional_feats, return_attention=True)
            # outputs => [batch_size, 1]
            # attn_list => list of attention tensors from each layer

            probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()  # Shape: [batch_size]
            preds = (probs >= 0.5).astype(int)                       # Shape: [batch_size]
            labels_np = labels.cpu().numpy()                          # Shape: [batch_size]

            # attn_list: List of tensors [batch_size, num_heads, seq_len, seq_len] or [batch_size, seq_len, seq_len]
            for i in range(stft_features.size(0)):
                sample_attentions = []
                for layer_attn in attn_list:
                    if layer_attn is not None:
                        # Handle multi-head attention if applicable
                        if layer_attn.ndim == 4:
                            # layer_attn: [batch_size, num_heads, seq_len, seq_len]
                            attn_matrix = layer_attn[i].cpu().numpy()  # Shape: [num_heads, seq_len, seq_len]
                            # Optionally, average over heads
                            attn_avg = attn_matrix.mean(axis=0)  # Shape: [seq_len, seq_len]
                            sample_attentions.append(attn_avg)
                        elif layer_attn.ndim == 3:
                            # layer_attn: [batch_size, seq_len, seq_len]
                            attn_matrix = layer_attn[i].cpu().numpy()  # Shape: [seq_len, seq_len]
                            sample_attentions.append(attn_matrix)
                        else:
                            # Unexpected dimension
                            sample_attentions.append(None)
                    else:
                        sample_attentions.append(None)

                attention_data.append({
                    'label': int(labels_np[i]),
                    'pred': int(preds[i]),
                    'prob': float(probs[i]),
                    'attentions': sample_attentions  # List of attention matrices per layer
                })

    return attention_data


# ==================================
# Extract Feature Importances
# ==================================

# Choose attribution method: 'saliency' or 'integrated_gradients'
method = 'saliency'  # or 'integrated_gradients'

# Extract attributions for the training set (stft_features)
train_attrib_stft, train_labels_stft, all_stft_inputs, all_procedure_times = extract_feature_importance_stft(
    model_wrapper_stft,
    analysis_loader,
    device,
    method=method
)

# Extract attributions for the training set (additional_feats)
train_attrib_extra, train_labels_extra = extract_feature_importance_extra(
    model_wrapper_extra,
    analysis_loader,
    device,
    method=method
)

# Extract attention weights from the training set
train_attention_data = extract_attention_weights(model, analysis_loader, device)

# Constants for the number of axes and frequency bins per axis
NUM_AXES = 7  # x, y, z, quat1, quat2, quat3, quat4
FREQ_BINS_PER_AXIS = 129

def separate_axes(saliency_maps):
    """
    Split the saliency maps into 7 axes with shape [7 * N, FREQ_BINS_PER_AXIS, TIME_FRAMES].

    Parameters:
        saliency_maps (np.ndarray): Saliency maps, shape [N, 903, TIME_FRAMES].

    Returns:
        np.ndarray: Combined saliency maps for all axes, shape [7 * N, FREQ_BINS_PER_AXIS, TIME_FRAMES].
    """
    separated_maps = []
    for i in range(NUM_AXES):
        start_idx = i * FREQ_BINS_PER_AXIS
        end_idx = (i + 1) * FREQ_BINS_PER_AXIS
        # Extract axis-specific maps
        separated_maps.append(saliency_maps[:, start_idx:end_idx, :])
    
    # Stack along the batch dimension
    combined_maps = np.concatenate(separated_maps, axis=0)  # Shape: [7 * N, FREQ_BINS_PER_AXIS, TIME_FRAMES]
    return combined_maps



def plot_saliency_multiplied_with_input(
    saliency_maps: np.ndarray,
    stfts: np.ndarray,
    sample_indices: List[int],
    time_range: Tuple[int, int] = None
):
    """
    Plot saliency maps multiplied with corresponding STFTs.

    Parameters:
        saliency_maps (np.ndarray): Saliency maps, shape [N, channels, time_frames].
        stfts (np.ndarray): STFT inputs, shape [N, channels, time_frames].
        sample_indices (List[int]): Indices of samples to visualize.
        time_range (Tuple[int, int], optional): Range of time frames to crop (start, end).
    """
    for idx in sample_indices:
        saliency_map = saliency_maps[idx]
        stft = stfts[idx]

        # Multiply saliency with input
        saliency_weighted_stft = saliency_map * stft  # Shape: (channels, time_frames)

        # Crop time range if specified
        if time_range:
            start, end = time_range
            saliency_weighted_stft = saliency_weighted_stft[:, start:end]

        # Plot the result
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            saliency_weighted_stft,  # Keep 2D array intact
            cmap="bwr",
            cbar=True
        )
        plt.title(f"Saliency Weighted STFT - Sample {idx}")
        plt.xlabel("Time Frames")
        plt.ylabel("Channels")
        plt.show()


plot_saliency_multiplied_with_input(saliency_maps=train_attrib_stft, stfts=all_stft_inputs, sample_indices=range(2), time_range=(0, 500))

def all_saliency_weighted_stft(saliency_maps, stfts, time_range=None):
    list = []
    for idx, stft in enumerate(stfts):
        saliency_map = saliency_maps[idx]

        # Multiply saliency with input
        saliency_weighted_stft = saliency_map * stft  # Shape: (channels, time_frames)
        list.append(saliency_weighted_stft)

    return np.array(list)

        
def overlay_saliency_weighted_inputs(
    saliency_maps: np.ndarray,
    stfts: np.ndarray,
    labels: np.ndarray,
    time_range: Tuple[int, int] = None
):
    """
    Multiply saliency maps with STFTs and overlay the results for positive, negative, and all samples.

    Parameters:
        saliency_maps (np.ndarray): Saliency maps, shape [N, channels, time_frames].
        stfts (np.ndarray): STFT inputs, shape [N, channels, time_frames].
        labels (np.ndarray): Binary labels for the samples, shape [N].
        time_range (Tuple[int, int], optional): Range of time frames to crop (start, end).
    """
    # Filter saliency maps and STFTs based on labels
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    # Multiply saliency maps with inputs
    weighted_inputs_pos = saliency_maps[pos_indices] * stfts[pos_indices]
    weighted_inputs_neg = saliency_maps[neg_indices] * stfts[neg_indices]
    weighted_inputs_all = saliency_maps * stfts

    # Crop time range if specified
    if time_range:
        start, end = time_range
        weighted_inputs_pos = weighted_inputs_pos[:, :, start:end]
        weighted_inputs_neg = weighted_inputs_neg[:, :, start:end]
        weighted_inputs_all = weighted_inputs_all[:, :, start:end]

    # Compute the mean across all samples for each set
    avg_weighted_pos = np.mean(weighted_inputs_pos, axis=0)
    avg_weighted_neg = np.mean(weighted_inputs_neg, axis=0)
    avg_weighted_all = np.mean(weighted_inputs_all, axis=0)

    # Plot results
    for avg_weighted, title in zip(
        [avg_weighted_pos, avg_weighted_neg, avg_weighted_all],
        ["Positive Samples", "Negative Samples", "All Samples"]
    ):
        plt.figure(figsize=(12, 6))
        plt.imshow(avg_weighted, aspect='auto', cmap='bwr')
        plt.colorbar(label='Saliency-Weighted Input')
        plt.title(f"Average Saliency-Weighted Input - {title}")
        plt.xlabel("Time Frames")
        plt.ylabel("Channels")
        plt.show()

overlay_saliency_weighted_inputs(
    saliency_maps=train_attrib_stft,
    stfts=all_stft_inputs,
    labels=train_labels_stft,
    time_range=(0, 240)
)

# Define a helper function for plotting summed saliency-weighted STFTs
def plot_summed_saliency(saliency_data, sum_axis, title, xlabel, ylabel):
    """
    Plots the sum of saliency-weighted STFTs along a specified axis.

    Parameters:
        saliency_data (np.ndarray): Saliency-weighted STFT data.
        sum_axis (int): Axis along which to sum (0 for frequency, 1 for time).
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    summed_saliency = np.sum(saliency_data, axis=sum_axis)  # Sum along the specified axis
    mean_saliency = np.mean(summed_saliency, axis=0)        # Average across samples

    plt.figure(figsize=(10, 6))
    plt.plot(mean_saliency)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

# Compute saliency-weighted STFTs for positive, negative, and all samples
saliency_weighted_stft = all_saliency_weighted_stft(saliency_maps=train_attrib_stft, stfts=all_stft_inputs)

positive_indices_stft = np.where(train_labels_stft == 1)[0]
negative_indices_stft = np.where(train_labels_stft == 0)[0]

saliency_weighted_stft_pos = (saliency_weighted_stft[positive_indices_stft])
saliency_weighted_stft_neg = saliency_weighted_stft[negative_indices_stft]
saliency_weighted_stft_all = saliency_weighted_stft


def reduce_frequency_resolution(saliency_maps, target_bins=100):
    """
    Reduce the frequency resolution of saliency maps to the target number of bins using interpolation.

    Parameters:
        saliency_maps (np.ndarray): Saliency-weighted STFT maps of shape [N, freq_bins, time_steps].
        target_bins (int): Number of frequency bins to reduce to.

    Returns:
        np.ndarray: Saliency maps with reduced frequency resolution [N, target_bins, time_steps].
    """
    original_bins = saliency_maps.shape[1]
    new_bin_indices = np.linspace(0, original_bins - 1, target_bins)

    # Interpolate along the frequency axis
    reduced_maps = np.array([
        np.array([
            np.interp(new_bin_indices, np.arange(original_bins), saliency_maps[sample_idx, :, time_idx])
            for time_idx in range(saliency_maps.shape[2])
        ]).T  # Transpose to get shape (target_bins, time_steps)
        for sample_idx in range(saliency_maps.shape[0])
    ])
    return reduced_maps


def normalize_by_procedure_time(saliency_maps, n_common_steps=240):
    """
    Normalize saliency maps by the duration of their non-zero signals.

    Parameters:
        saliency_maps (np.ndarray): Saliency-weighted STFT maps of shape [N, 903, time_steps].
        n_common_steps (int): Number of common steps to interpolate to.

    Returns:
        np.ndarray: Normalized saliency maps of shape [N, 903, n_common_steps].
    """
    if not isinstance(n_common_steps, int):  # Ensure n_common_steps is an integer
        raise ValueError("n_common_steps must be an integer.")

    normalized_maps = []

    for saliency_map in saliency_maps:
        # Compute effective length based on non-zero time frames
        non_zero_indices = np.where(np.sum(np.abs(saliency_map), axis=0) != 0)[0]
        
        if len(non_zero_indices) == 0:
            effective_steps = 1  # Handle edge case of completely zero maps
            truncated_map = saliency_map[:, :effective_steps]
        else:
            start_idx, end_idx = non_zero_indices[0], non_zero_indices[-1] + 1
            effective_steps = end_idx - start_idx
            truncated_map = saliency_map[:, start_idx:end_idx]  # Truncate to non-zero region

        # Create normalized time axis
        original_time = np.linspace(0, 1, effective_steps)
        new_time = np.linspace(0, 1, n_common_steps)

        # Interpolate saliency map to normalized time
        rescaled_map = np.array([
            np.interp(new_time, original_time, truncated_map[channel])
            for channel in range(truncated_map.shape[0])
        ])

        normalized_maps.append(rescaled_map)

    return np.array(normalized_maps)




def plot_normalized_saliency(normalized_neg_maps, normalized_pos_maps):
    """
    Plot the normalized saliency map summed along either frequency or time.

    Parameters:
        normalized_maps (np.ndarray): Array of normalized saliency maps.
        title (str): Plot title.
    """
    # Sum along frequency and time axes
    summed_time_neg = np.sum(normalized_neg_maps, axis=1)  # (samples, time)
    summed_freq_neg = np.sum(normalized_neg_maps, axis=2)  # (samples, freq)

    avg_time_saliency_neg = np.mean(summed_time_neg, axis=0)
    avg_freq_saliency_neg = np.mean(summed_freq_neg, axis=0)

    summed_time_pos = np.sum(normalized_pos_maps, axis=1)  # (samples, time)
    summed_freq_pos = np.sum(normalized_pos_maps, axis=2)  # (samples, freq)

    avg_time_saliency_pos = np.mean(summed_time_pos, axis=0)
    avg_freq_saliency_pos = np.mean(summed_freq_pos, axis=0)

    # Plot summed along time
    plt.figure(figsize=(10, 4))
    plt.plot(avg_time_saliency_pos, label='Positive (Label=1)', color='red')
    plt.plot(avg_time_saliency_neg, label='Negative (Label=0)', color='blue')
    plt.title(f'Temporal importance')
    plt.xlabel('0 = start of procedure, 100 = end of procedure')
    plt.ylabel('Average attribution over all samples')
    plt.ylim([-2e-5, 7e-5])
    plt.legend()
    plt.show()

    # Plot summed along frequency
    plt.figure(figsize=(10, 4))
    plt.plot(avg_freq_saliency_pos, label='Positive (Label=1)', color='red')
    plt.plot(avg_freq_saliency_neg, label='Negative (Label=0)', color='blue')
    plt.title(f'Spectral importance')
    plt.xlabel('Frequency bins')
    plt.ylabel('Average attribution over all samples')
    plt.ylim([-2e-5, 10e-5])
    plt.legend()
    plt.show()

# Reduce frequency resolution before normalization
reduced_saliency_weighted_stft_pos = reduce_frequency_resolution(separate_axes(saliency_weighted_stft_pos), target_bins=100)
reduced_saliency_weighted_stft_neg = reduce_frequency_resolution(separate_axes(saliency_weighted_stft_neg), target_bins=100)
reduced_saliency_weighted_stft_all = reduce_frequency_resolution(saliency_weighted_stft_all, target_bins=100)

# Normalize maps by procedure time (based on non-zero signals)
normalized_pos_maps = normalize_by_procedure_time(reduced_saliency_weighted_stft_pos, n_common_steps=100)
normalized_neg_maps = normalize_by_procedure_time(reduced_saliency_weighted_stft_neg, n_common_steps=100)
normalized_all_maps = normalize_by_procedure_time(reduced_saliency_weighted_stft_all, n_common_steps=100)


# Plot normalized maps
plot_normalized_saliency(normalized_neg_maps, normalized_pos_maps)
