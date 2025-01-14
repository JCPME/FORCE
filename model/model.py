# %%
import pandas as pd
import numpy as np
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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm


warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

"""
Installation Instructions:
--------------------------
You can install the required packages using pip. For example:

    pip install -r requirements.txt

Or conda (if you are using Anaconda distribution):
    conda install requirements.txt

If you are running this script within a Jupyter Notebook, ensure that your kernel is using the correct Python version and has these packages installed.

Additional Notes:
    - If you are utilizing GPU acceleration, make sure PyTorch is installed with CUDA support.
    - Verify that your data file (e.g., 'dataset.pkl') is accessible at ../data/dataset.pkl.

--------------------------
"""



# %%
# ==================================
# Set Seed for Reproducibility
# ==================================
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================================
# Helper Functions for Quaternions and Augmentation functions
# ==================================
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)

def euler_to_quaternion(euler):
    roll, pitch, yaw = euler.unbind(-1)

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)

def rotate_points_by_quaternion(points, quaternion):
    point_quats = torch.cat([torch.zeros_like(points[..., :1]), points], dim=-1)
    quaternion_conjugate = quaternion * torch.tensor([1, -1, -1, -1], device=quaternion.device)
    rotated = quaternion_multiply(quaternion_multiply(quaternion, point_quats), quaternion_conjugate)
    return rotated[..., 1:]

def add_gaussian_noise(data, quat_sigma=0.001, pos_sigma=0.002):
    quats = data[..., 3:]
    pos = data[..., :3]

    quat_noise = torch.randn_like(quats) * quat_sigma
    noisy_quats = quats + quat_noise
    noisy_quats = F.normalize(noisy_quats, dim=-1)

    pos_noise = torch.randn_like(pos) * pos_sigma
    noisy_pos = pos + pos_noise

    return torch.cat([noisy_pos, noisy_quats], dim=-1)

def spatial_rotation(data, max_angle=80, min_angle=60):
    device = data.device
    batch_size = data.shape[0]

    random_angles = torch.rand(batch_size, 3, device=device) * (max_angle - min_angle) + min_angle
    random_sign = torch.sign(torch.rand(batch_size, 3, device=device) - 0.5)
    random_angles *= random_sign
    rotation_quat = euler_to_quaternion(random_angles)

    quats = data[..., 3:]
    pos = data[..., :3]

    rotated_quats = quaternion_multiply(rotation_quat.unsqueeze(1).expand(-1, quats.shape[1], -1), quats)
    rotated_pos = rotate_points_by_quaternion(pos, rotation_quat.unsqueeze(1))

    return torch.cat([rotated_pos, rotated_quats], dim=-1)

def mirror_trajectory(data, axis=2):
    quats = data[..., 3:]
    pos = data[..., :3]

    reflection = torch.ones_like(pos)
    reflection[..., axis] *= -1
    mirrored_pos = pos * reflection

    mirrored_quats = quats.clone()
    mirrored_quats[..., axis] *= -1
    mirrored_quats = F.normalize(mirrored_quats, dim=-1)

    return torch.cat([mirrored_pos, mirrored_quats], dim=-1)

def combine_translation_rotation(row):
    """
    Combine translation and rotation arrays into a single 7D array.

    Parameters:
        row (pd.Series): A row from the DataFrame.

    Returns:
        np.ndarray: Combined 7D array of shape (N, 7).
    """
    translation = np.array(row['translation_array'])  # Shape: (N, 3)
    rotation    = np.array(row['rotation_array'])     # Shape: (N, 4)

    if len(translation) != len(rotation):
        raise ValueError("Translation and rotation arrays must have the same length.")

    combined = np.hstack((translation, rotation))     # Shape: (N, 7)
    return combined

def augment_data(combined_array, num_augmentations=7):
    """
    Augment a single combined 7D array using the provided helper functions.

    Parameters:
        combined_array (np.ndarray): Original 7D array of shape (N, 7).
        num_augmentations (int): Number of augmented samples to generate.

    Returns:
        List[np.ndarray]: List containing the original and augmented 7D arrays.
    """
    data_torch = torch.tensor(combined_array, dtype=torch.float32)  # Shape: (N, 7)

    augmented_torch = augment_kinematic_data(data_torch.unsqueeze(0), num_augmentations=num_augmentations)  # Shape: (1 + num_augmentations, N, 7)

    augmented_np = augmented_torch

    augmented_samples = [augmented_np[i] for i in range(augmented_np.shape[0])]

    return augmented_samples

def augment_kinematic_data(data, num_augmentations=7):
    data_t = torch.tensor(data, dtype=torch.float32)
    augmentation_functions = [
        spatial_rotation,
        mirror_trajectory,
        add_gaussian_noise
    ]

    augmented_data = [data_t]
    for _ in range(num_augmentations):
        aug_func = random.choice(augmentation_functions)
        augmented_data.append(aug_func(data_t.clone()))

    return torch.cat(augmented_data, dim=0).cpu().numpy()

# %%
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
        return np.zeros((7 * (n_fft // 2 + 1), fixed_time_frames), dtype=np.float32)

    translation_array = np.array(translation_array)  # (n, 3)
    rotation_array = np.array(rotation_array)        # (n, 4)

    t0, t_end = timestamp_array[0], timestamp_array[-1]
    new_t = np.linspace(t0, t_end, n)

    trans = np.zeros((n, 3))
    rot   = np.zeros((n, 4))

    for i in range(3):
        trans[:, i] = np.interp(new_t, timestamp_array, translation_array[:, i])
    for i in range(4):
        rot[:, i]   = np.interp(new_t, timestamp_array, rotation_array[:, i])

    trans_torch = torch.tensor(trans, dtype=torch.float32)  # shape: (n, 3)
    rot_torch   = torch.tensor(rot,   dtype=torch.float32)  # shape: (n, 4)

    def stft_mag(signal_1d):
        if signal_1d.size(0) < n_fft:
            pad_size = n_fft - signal_1d.size(0)
            signal_1d = F.pad(signal_1d, (0, pad_size), 'constant', 0)

        window = torch.hann_window(n_fft)
        stft_res = torch.stft(
            signal_1d,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            return_complex=False
        )

        real_part = stft_res[..., 0]
        imag_part = stft_res[..., 1]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        return magnitude  

    all_mags = []

    for i in range(3):
        mag = stft_mag(trans_torch[:, i])  # shape: (freq_bins, time_frames)
        all_mags.append(mag.unsqueeze(0))

    # rotation => w,x,y,z
    for i in range(4):
        mag = stft_mag(rot_torch[:, i])    # shape: (freq_bins, time_frames)
        all_mags.append(mag.unsqueeze(0))

    cat_mags = torch.cat(all_mags, dim=0)  # shape: [7, freq_bins, time_frames]

    stft_2d = cat_mags.view(7 * (n_fft//2 + 1), -1).numpy()  # shape: [7*(freq_bins), time_frames]

    current_time_frames = stft_2d.shape[1]
    if current_time_frames < fixed_time_frames:
        padding = fixed_time_frames - current_time_frames
        stft_2d = np.pad(stft_2d, ((0, 0), (0, padding)), mode='constant')
    elif current_time_frames > fixed_time_frames:
        stft_2d = stft_2d[:, :fixed_time_frames]

    return stft_2d.astype(np.float32)

# %%
# ==================================
# Data Loading
# ==================================
data_path = './data/dataset.pkl'
df = pd.read_pickle(data_path)

print("Columns in DataFrame:", df.columns)


# %%
# ------------------------------
# 2) Create Binary Label (above/below mean avg_grs_score)
# ------------------------------

# Create binary label based on average GRS score split by participants and tools
mean_score = df['avg_grs_score'].mean()
df['label'] = (df['avg_grs_score'] >= mean_score).astype(int)
unique_participants = df['participant_num'].unique()


unique_participants = unique_participants.tolist()
random.shuffle(unique_participants)

n_total = len(unique_participants)
n_test  = int(0.2 * n_total)
n_val   = int(0.2 * n_total)
n_train = n_total - n_test - n_val

test_participants = unique_participants[:n_test]
val_participants  = unique_participants[n_test:n_test + n_val]
train_participants = unique_participants[n_test + n_val:]

tools_to_include = ['Ulna_1', 'Ulna_2', 'Radius_1', 'Radius_2']
filtered_df = df[df['tool'].isin(tools_to_include)]
print("\nFiltered DataFrame Description:")
print(filtered_df.describe())

train_df = filtered_df[filtered_df['participant_num'].isin(train_participants)]
val_df   = filtered_df[filtered_df['participant_num'].isin(val_participants)]
test_df  = filtered_df[filtered_df['participant_num'].isin(test_participants)]

print(f"\nTrain: {train_df.shape[0]} rows, Participants: {len(train_participants)}")
print(f"Validation: {val_df.shape[0]} rows, Participants: {len(val_participants)}")
print(f"Test: {test_df.shape[0]} rows, Participants: {len(test_participants)}")

# ==================================
# Data Preprocessing and Augmentation
# ==================================
augment_data_flag = True  


def augment_data_in_batches(df, batch_size=128, num_augmentations=7):
    augmented_data = []
    augmented_labels = []
    augmented_tool = []
    augmented_case = []
    augmented_procedure_time = []

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch_df = df.iloc[start:end]
        for idx, row in batch_df.iterrows():
            original_combined = row['combined_array']
            label = row['label']
            tool = row['tool']
            case = row['case']
            total_time = row['total_procedure_time']

            augmented_samples = augment_data(original_combined, num_augmentations=num_augmentations)

            for sample in augmented_samples:
                augmented_data.append(sample)
                augmented_labels.append(label)
                augmented_tool.append(tool)
                augmented_case.append(case)
                augmented_procedure_time.append(total_time)

    new_augmented_df = pd.DataFrame({
        'participant_num': np.repeat(df['participant_num'].values, 8),
        'tool': np.repeat(df['tool'].values, 8),
        'case': np.repeat(df['case'].values, 8),
        'translation_array': [sample[:, :3] for sample in augmented_data],
        'rotation_array': [sample[:, 3:] for sample in augmented_data],
        'timestamp_array': np.repeat(df['timestamp_array'].values, 8),
        'avg_grs_score': np.repeat(df['avg_grs_score'].values, 8),
        'total_procedure_time': np.repeat(df['total_procedure_time'].values, 8),
        'label': augmented_labels
    })
    return new_augmented_df


# ==================================
# Augment Data and Balance Validation Set
# ==================================


if augment_data_flag:
    train_df = train_df.copy() 
    train_df['combined_array'] = train_df.apply(combine_translation_rotation, axis=1)

    augmented_df = augment_data_in_batches(train_df, batch_size=128, num_augmentations=7)

    print(f"\nOriginal Train Set: {train_df.shape[0]} rows")
    print(f"Augmented Train Set: {augmented_df.shape[0]} rows")

    print("\nAugmented DataFrame Head:")
    print(augmented_df.head())
    print("\nAugmented DataFrame Description:")
    print(augmented_df.describe())
    print("\nLabel Distribution in Augmented DataFrame:")
    print(augmented_df['label'].value_counts())

    # Augment Validation Set
    val_df = val_df.copy()  # To avoid SettingWithCopyWarning
    val_df['combined_array'] = val_df.apply(combine_translation_rotation, axis=1)
    augmented_val_df = augment_data_in_batches(val_df, batch_size=128, num_augmentations=7)

    # Balance the Augmented Validation Set
    from sklearn.utils import resample

    # Separate majority and minority classes
    majority = augmented_val_df[augmented_val_df['label'] == augmented_val_df['label'].value_counts().idxmax()]
    minority = augmented_val_df[augmented_val_df['label'] == augmented_val_df['label'].value_counts().idxmin()]

    # Upsample minority class
    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=42)

    # Combine majority and upsampled minority
    augmented_val_df_balanced = pd.concat([majority, minority_upsampled])

    val_df = augmented_val_df_balanced

    print(f"\nOriginal Validation Set: {val_df.shape[0]} rows")
    print(f"Augmented and Balanced Validation Set: {val_df.shape[0]} rows")

    print("\nAugmented Validation DataFrame Head:")
    print(augmented_val_df_balanced.head())
    print("\nAugmented Validation DataFrame Description:")
    print(augmented_val_df_balanced.describe())
    print("\nLabel Distribution in Augmented Validation DataFrame:")
    print(augmented_val_df_balanced['label'].value_counts())
else:
    augmented_df = train_df

# %%
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
    """
    Constructs feature matrices using STFT features and additional numeric features.
        input_df: DataFrame with augmented data
        tool_encoder: LabelEncoder object for tools
        case_encoder: LabelEncoder object for cases
        n_fft: Number of FFT components
        hop_length: Number of samples between successive frames

    Returns:
        X_list: List of tuples containing (stft_2d, extra_feats)
        y_list: List of labels
        tool_encoder: Fitted LabelEncoder for tools
        case_encoder: Fitted LabelEncoder
    """

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
    Constructs feature matrices using STFT features and additional numeric features for the testset.
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
X_train, y_train, tool_encoder, case_encoder = build_feature_matrix(
    augmented_df,
    tool_encoder=None,
    case_encoder=None,
    n_fft=n_fft,
    hop_length=hop_length
)

print("\nNumber of Augmented Train Samples:", len(X_train))
print("Number of Augmented Train Labels:", y_train.shape)

stft_train = np.array([x[0] for x in X_train])  # Shape: (num_samples, channels, time_frames)
extra_train = np.array([x[1] for x in X_train]) # Shape: (num_samples, 3)

scaler_stft = StandardScaler()
scaler_extra = StandardScaler()

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

X_train_scaled = list(zip(stft_train_scaled, extra_train_scaled))

X_val_scaled, y_val = build_feature_matrix_test(
    val_df,
    tool_encoder=tool_encoder,
    case_encoder=case_encoder,
    scaler_stft=scaler_stft,
    scaler_extra=scaler_extra,
    n_fft=n_fft,
    hop_length=hop_length
)

X_test_scaled, y_test = build_feature_matrix_test(
    test_df,
    tool_encoder=tool_encoder,
    case_encoder=case_encoder,
    scaler_stft=scaler_stft,
    scaler_extra=scaler_extra,
    n_fft=n_fft,
    hop_length=hop_length
)

print("\nValidation Set Size:", len(X_val_scaled))
print("Test Set Size:", len(X_test_scaled))

# %%
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
        self.fixed_time_frames = 1500  # Ensure consistency with padding

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        stft_2d, extra_feats = self.X[idx]
        # Create mask: 1 for valid frames, 0 for padded
        mask = (stft_2d.sum(axis=0) != 0).astype(float)
        # Convert to torch tensors
        stft_2d = torch.tensor(stft_2d, dtype=torch.float32)
        extra_feats = torch.tensor(extra_feats, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        label = self.y[idx]
        return stft_2d, extra_feats, mask, label

# %%
# ==================================
# Create Datasets and DataLoaders
# ==================================

train_dataset = FFTAdditionalDataset(X_train_scaled, y_train)
val_dataset   = FFTAdditionalDataset(X_val_scaled, y_val)
test_dataset  = FFTAdditionalDataset(X_test_scaled, y_test)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("\nDataLoaders Created:")
print(f"Train Loader: {len(train_loader)} batches")
print(f"Validation Loader: {len(val_loader)} batches")
print(f"Test Loader: {len(test_loader)} batches")

# %%
# ==================================
# Define Model Components
# ==================================
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
            out, attn_weights = self.network[0](x, return_attention=True)
            for layer in self.network[1:]:
                out, attn_weights = layer(out, return_attention=True)
            return out, attn_weights
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
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_sizes[0], ffn_hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.ffn_output_size = ffn_hidden_sizes[-1]

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.tcn_output_size + self.ffn_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, stft_features, additional_features, mask, return_attention=False):
        """
        stft_features: [batch_size, channels, time_frames]
        additional_features: [batch_size, 3]
        mask: [batch_size, time_frames]
        """
        # Apply mask by multiplying stft_features with mask
        mask = mask.unsqueeze(1)  # [batch_size, 1, time_frames]
        stft_features = stft_features * mask

        tcn_out = self.tcn(stft_features)
        tcn_out = torch.mean(tcn_out, dim=2)  # [batch_size, out_channels]

        ffn_out = self.ffn(additional_features)  # [batch_size, ffn_output_size]

        combined = torch.cat((tcn_out, ffn_out), dim=1)  # [batch_size, out_channels + ffn_output_size]
        out = self.classifier(combined)  # [batch_size, 1]

        if return_attention:
            return out, attention
        return out

# %%
# ==================================
# Initialize the Model
# ==================================
# Calculate channels based on n_fft
freq_bins = n_fft // 2 + 1  # 129 for n_fft=256
channels = 7 * freq_bins    # 7 axes (3 translation + 4 rotation)

# Initialize the model
model = TCN_FFN_Model(
    tcn_input_channels=channels,        # 7 * 129 = 903
    num_channels=[64, 64, 64],
    kernel_size=3,
    dropout=0, # removed dropout for best results
    additional_input_size=3,
    ffn_hidden_sizes=[64, 32],
    use_attention=True,
    n_heads=1
)

model.to(device)
print("\nModel Initialized and Moved to Device.")

# %%
# ==================================
# Define Loss Function and Optimizer
# ==================================

label_counts = augmented_df['label'].value_counts().to_dict()
total_samples = len(augmented_df)
weight_neg = total_samples / (2 * label_counts[0])
weight_pos = total_samples / (2 * label_counts[1])

class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(device)

def weighted_bce_loss(outputs, targets):
    weights = targets * class_weights[1] + (1 - targets) * class_weights[0]
    return torch.nn.functional.binary_cross_entropy(outputs, targets, weight=weights)

criterion = weighted_bce_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Adjusted learning rate

# Define Early Stopping parameters
patience = 10  
best_val_loss = float('inf')
epochs_no_improve = 0
n_epochs = 100  

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


epoch_progress = tqdm(range(n_epochs), desc="Training", position=0)

for epoch in epoch_progress:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0


    batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training", leave=False, position=1)
    for stft_features, additional_feats, mask, labels in batch_progress:
        stft_features = stft_features.to(device)           # [batch_size, channels, time_frames]
        additional_feats = additional_feats.to(device)     # [batch_size, 3]
        mask = mask.to(device)                             # [batch_size, time_frames]
        labels = labels.to(device).unsqueeze(1)           # [batch_size, 1]

        optimizer.zero_grad()

        outputs = model(stft_features, additional_feats, mask)    # [batch_size, 1]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * stft_features.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        batch_progress.set_postfix({'Batch Loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        val_batch_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Validation", leave=False, position=1)
        for stft_features, additional_feats, mask, labels in val_batch_progress:
            stft_features = stft_features.to(device)
            additional_feats = additional_feats.to(device)
            mask = mask.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(stft_features, additional_feats, mask)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * stft_features.size(0)
            preds = (outputs >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            # Optionally, update val_batch_progress with batch loss
            val_batch_progress.set_postfix({'Val Batch Loss': f"{loss.item():.4f}"})

    val_epoch_loss = val_running_loss / val_total
    val_epoch_acc = val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    epoch_progress.set_postfix({
        'Train Loss': f"{epoch_loss:.4f}",
        'Train Acc': f"{epoch_acc:.4f}",
        'Val Loss': f"{val_epoch_loss:.4f}",
        'Val Acc': f"{val_epoch_acc:.4f}"
    })

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_tcn_ffn_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            epoch_progress.write("Early stopping!")
            break


epoch_progress.close()

# %%
# ==================================
# Model Evaluation on Test Set
# ==================================
# Load the best model
model.load_state_dict(torch.load('best_tcn_ffn_model.pth'))
model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for stft_features, additional_feats, mask, labels in tqdm(test_loader, desc="Test Evaluation"):
        stft_features = stft_features.to(device)
        additional_feats = additional_feats.to(device)
        mask = mask.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(stft_features, additional_feats, mask)
        probs = outputs.cpu().numpy().flatten()
        preds = (outputs >= 0.5).float().cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)

# Classification Report
print("\n Test Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Below Mean', 'Above Mean']))

# ROC-AUC Score
roc_auc = roc_auc_score(all_labels, all_probs)
print(f"Test ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)



fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()



# %%
# ==================================
# Training Progress Visualization (Optional)
# ==================================
# Plot Training and Validation Loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot Training and Validation Accuracy
plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


