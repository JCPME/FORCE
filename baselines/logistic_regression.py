import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import random
import warnings
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore")

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


data_path = 'data/output_with_bones.pkl'
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

def train_simple_model_on_seq_length(
    df: pd.DataFrame,
    translation_col: str = 'translation_array',
    label_col: str = 'label',
    test_size: float = 0.2,
    random_state: int = 42
):

    # 1) Compute sequence length
    df = df.copy()
    df['seq_length'] = df[translation_col].apply(lambda arr: len(arr))

    # 2) Prepare X and y
    X = df[['seq_length']].values  # shape: (N, 1)
    y = df[label_col].values       # shape: (N,)

    # 3) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # preserve label distribution
    )

    # 4) Train a simple logistic regression on just this one feature
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.4f}\n")

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Label=0', 'Label=1']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 6) ROC-AUC computation
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # 7) ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.grid()
    plt.show()

    return model, (X_test, y_test)


simple_model, (X_test, y_test) = train_simple_model_on_seq_length(filtered_df)
y_pred = simple_model.predict(X_test)