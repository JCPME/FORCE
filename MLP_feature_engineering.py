import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataContainer:
    def __init__(self, transformation, timestamp):
        self.transformation = transformation
        self.timestamp = timestamp

class TransformationContainer:
    def __init__(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation

def load_surgical_data(data_path):
    
    df = pd.read_pickle(data_path)

    recordings = []
    labels = []

    for idx, row in df.iterrows():
        try:
            if ("Ulna" not in row["tool"]) and ("Radius" not in row["tool"]):
                continue
            translations = np.array(row['translation_array'])
            rotations = np.array(row['rotation_array'])
            timestamps = np.array(row['timestamp_array'])

            transform = TransformationContainer(
                translation=translations,
                rotation=rotations
            )

            recording = DataContainer(
                transformation=transform,
                timestamp=timestamps
            )

            recordings.append(recording)
            labels.append(row['avg_grs_score'])

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    return recordings, np.array(labels)




data_path = "dataset.pkl"
recordings_unfiltered, scores_unfiltered = load_surgical_data(data_path)

print(f"Loaded {len(recordings_unfiltered)} total recordings")


recordings = []
labels = []
for i in range(len(recordings_unfiltered)):
    if (len(recordings_unfiltered[i].timestamp) > 50):
      recordings.append(recordings_unfiltered[i])
      labels.append(scores_unfiltered[i])

print(f"Loaded {len(recordings)} recordings with sequence length > 50")

import numpy as np
from scipy import stats
#from helpers import *
from scipy.stats import entropy
from scipy.signal import butter, filtfilt
from scipy.spatial import ConvexHull
import traceback



def calculate_time_per_activity( recordings, i, activities):
    timestamps = recordings[i].timestamp
    times = np.zeros(len(activities))

    for idx, (start_time, end_time) in enumerate(activities):
        start_idx = np.searchsorted(timestamps.astype(int), int(start_time * 1000), side='left')
        end_idx = np.searchsorted(timestamps.astype(int), int(end_time * 1000), side='right') - 1

        activity_timestamps = timestamps[start_idx:end_idx + 1]
        times[idx] = (activity_timestamps[-1] - activity_timestamps[0]) / 1000.0  # Convert ms to seconds

    return times

def calculate_translational_path_and_speed(recording):
   
    translations = recording.transformation.translation
    timestamps = recording.timestamp

    if len(translations) < 2 or len(timestamps) < 2:
        return 0.0, 0.0

    diffs = np.diff(translations, axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    time_deltas = np.diff(timestamps) / 1000.0

    valid_indices = np.where((time_deltas > 0) & (time_deltas <= 0.1))[0]
    filtered_distances = distances[valid_indices]
    filtered_time_deltas = time_deltas[valid_indices]

    velocities = filtered_distances / filtered_time_deltas
    valid_velocities = velocities[velocities <= 10]  # Filter out velocities > 10 m/s

    path_length = np.sum(filtered_distances[velocities <= 10])
    avg_velocity = np.mean(valid_velocities) if len(valid_velocities) > 0 else 0

    return path_length, avg_velocity

def calculate_standard_deviation_of_velocity(recording):
    
    translations = recording.transformation.translation
    timestamps = recording.timestamp

    if len(translations) < 2:
        return 0.0

    diffs = np.diff(translations, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    time_deltas = np.diff(timestamps) / 1000.0  # Convert to seconds

    valid_indices = np.where(time_deltas <= 0.1)[0]
    filtered_distances = distances[valid_indices]
    filtered_time_deltas = time_deltas[valid_indices]

    velocities = filtered_distances / filtered_time_deltas
    valid_velocities = velocities[velocities <= 10]

    return np.std(valid_velocities) if len(valid_velocities) > 0 else 0

def calculate_velocity_frequency_centroids(recording):
    """Calculate frequency centroids of velocity data"""
    translations = recording.transformation.translation
    timestamps = recording.timestamp

    if len(translations) < 2:
        return 0.0

    diffs = np.diff(translations, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    time_deltas = np.diff(timestamps) / 1000.0

    valid_indices = np.where(time_deltas <= 0.1)[0]
    filtered_distances = distances[valid_indices]
    filtered_time_deltas = time_deltas[valid_indices]

    velocities = filtered_distances / filtered_time_deltas
    valid_velocities = velocities[velocities <= 10]

    if len(valid_velocities) < 2:
        return 0.0

    sampling_rate = 1 / np.mean(filtered_time_deltas)
    fft_result = np.fft.fft(valid_velocities)
    frequencies = np.fft.fftfreq(len(valid_velocities), d=1/sampling_rate)

    power_spectral_density = np.abs(fft_result) ** 2

    positive_freq_indices = frequencies > 0
    positive_frequencies = frequencies[positive_freq_indices]
    positive_psd = power_spectral_density[positive_freq_indices]

    if np.sum(positive_psd) > 0:
        frequency_centroid = np.sum(positive_frequencies * positive_psd) / np.sum(positive_psd)
        return frequency_centroid
    return 0.0

def calculate_velocity_entropy(recording, num_bins=30):
    """Calculate entropy of velocity distribution"""
    translations = recording.transformation.translation
    timestamps = recording.timestamp

    if len(translations) < 2:
        return 0.0

    diffs = np.diff(translations, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    time_deltas = np.diff(timestamps) / 1000.0

    valid_indices = np.where(time_deltas <= 0.1)[0]
    filtered_distances = distances[valid_indices]
    filtered_time_deltas = time_deltas[valid_indices]

    velocities = filtered_distances / filtered_time_deltas
    valid_velocities = velocities[velocities <= 10]

    if len(valid_velocities) < 2:
        return 0.0

    histogram, _ = np.histogram(valid_velocities, bins=num_bins, density=True)
    probabilities = histogram / np.sum(histogram)
    non_zero_probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(non_zero_probabilities * np.log(non_zero_probabilities))

    return entropy


def calculate_rotational_path_and_speed(recording):
    """Calculate rotational path and speed"""
    rotations = recording.transformation.rotation
    timestamps = recording.timestamp

    if len(rotations) < 2:
        return 0.0, 0.0

    max_angular_velocity = 8 * np.pi  # Maximum allowed angular velocity (rad/s)
    total_angular_change = 0
    total_time_interval = 0

    for j in range(len(rotations) - 1):
        q1 = rotations[j]
        q2 = rotations[j + 1]
        delta_time = (timestamps[j + 1] - timestamps[j]) / 1000.0

        if delta_time <= 0 or delta_time > 0.1:
            continue

        # Ensure quaternions are in the same hemisphere
        if np.dot(q1, q2) < 0:
            q2 = -q2

        angle = 2 * np.arccos(np.clip(np.dot(q1, q2), -1.0, 1.0))
        angular_velocity = angle / delta_time

        if angular_velocity <= max_angular_velocity:
            total_angular_change += angle
            total_time_interval += delta_time

    avg_speed = total_angular_change / total_time_interval if total_time_interval > 0 else 0
    return total_angular_change, avg_speed

def calculate_rotational_speed_std(recording):
    """Calculate standard deviation of rotational speed"""
    rotations = recording.transformation.rotation
    timestamps = recording.timestamp

    if len(rotations) < 2:
        return 0.0

    max_angular_velocity = 8 * np.pi
    rotational_speeds = []

    for j in range(len(rotations) - 1):
        q1 = rotations[j]
        q2 = rotations[j + 1]
        delta_time = (timestamps[j + 1] - timestamps[j]) / 1000.0

        if delta_time <= 0 or delta_time > 0.1:
            continue

        if np.dot(q1, q2) < 0:
            q2 = -q2

        angle = 2 * np.arccos(np.clip(np.dot(q1, q2), -1.0, 1.0))
        angular_velocity = angle / delta_time

        if angular_velocity <= max_angular_velocity:
            rotational_speeds.append(angular_velocity)

    return np.std(rotational_speeds) if rotational_speeds else 0

def calculate_rotational_frequency_centroids(recording):
    """Calculate frequency centroids for rotational data"""
    rotations = recording.transformation.rotation
    timestamps = recording.timestamp

    if len(rotations) < 2:
        return 0.0

    angular_velocities = []
    max_angular_velocity = 8 * np.pi

    for j in range(len(rotations) - 1):
        q1 = rotations[j]
        q2 = rotations[j + 1]
        delta_time = (timestamps[j + 1] - timestamps[j]) / 1000.0

        if delta_time <= 0 or delta_time > 0.1:
            continue

        if np.dot(q1, q2) < 0:
            q2 = -q2

        angle = 2 * np.arccos(np.clip(np.dot(q1, q2), -1.0, 1.0))
        angular_velocity = angle / delta_time

        if angular_velocity <= max_angular_velocity:
            angular_velocities.append(angular_velocity)

    if len(angular_velocities) < 2:
        return 0.0

    # Frequency analysis
    sampling_rate = 1 / np.mean(np.diff(timestamps)) * 1000  # Convert to Hz
    fft_result = np.fft.fft(angular_velocities)
    frequencies = np.fft.fftfreq(len(angular_velocities), d=1/sampling_rate)

    power_spectral_density = np.abs(fft_result) ** 2

    positive_indices = frequencies > 0
    positive_frequencies = frequencies[positive_indices]
    positive_psd = power_spectral_density[positive_indices]

    if np.sum(positive_psd) > 0:
        centroid = np.sum(positive_frequencies * positive_psd) / np.sum(positive_psd)
        return centroid
    return 0.0



def calculate_accumulated_turning_angle(recordings, i,activities):
    rotations = recordings[i].transformation.rotation  # Assuming rotation is stored here as quaternions or angles
    timestamps = recordings[i].timestamp
    turning_angles = np.zeros(len(activities))

    for idx, (start_time, end_time) in enumerate(activities):
        start_idx = np.searchsorted(timestamps.astype(int), int(start_time * 1000), side='left')
        end_idx = np.searchsorted(timestamps.astype(int), int(end_time * 1000), side='right') - 1

        activity_rotations = rotations[start_idx:end_idx + 1, :]
        diffs = np.diff(activity_rotations, axis=0)
        angles = np.linalg.norm(diffs, axis=1)
        turning_angles[idx] = np.sum(angles)

    return turning_angles

from scipy.signal import butter, filtfilt

def calculate_point_density_filtered(recording, max_frequency=10, sampling_rate=100):
    """Calculate filtered point density with handling for short sequences"""
    translations = recording.transformation.translation

    if len(translations) < 2:
        return 0.0

    # For very short sequences, skip filtering and calculate density directly
    if len(translations) <= 15:
        centroid = translations.mean(axis=0)
        distances = np.linalg.norm(translations - centroid, axis=1)
        average_distance = np.mean(distances)
        return 1 / average_distance if average_distance > 0 else 0

    try:
        # Design a lower-order filter for shorter sequences
        nyquist_frequency = 0.5 * sampling_rate
        cutoff = max_frequency / nyquist_frequency

        # Adjust filter order based on sequence length
        filter_order = min(4, len(translations) // 4 - 1)
        if filter_order < 1:
            filter_order = 1

        b, a = butter(N=filter_order, Wn=cutoff, btype='low', analog=False)

        # Apply filter to each coordinate
        filtered_data = np.zeros_like(translations)
        for axis in range(translations.shape[1]):
            try:
                filtered_data[:, axis] = filtfilt(b, a, translations[:, axis])
            except Exception as e:
                # If filtering fails, use original data for this axis
                filtered_data[:, axis] = translations[:, axis]

        # Calculate point density
        centroid = filtered_data.mean(axis=0)
        distances = np.linalg.norm(filtered_data - centroid, axis=1)
        average_distance = np.mean(distances)

        return 1 / average_distance if average_distance > 0 else 0

    except Exception as e:
        print(f"Error in point density calculation: {e}")
        # Fallback to unfiltered calculation
        centroid = translations.mean(axis=0)
        distances = np.linalg.norm(translations - centroid, axis=1)
        average_distance = np.mean(distances)
        return 1 / average_distance if average_distance > 0 else 0


def calculate_rotational_entropy(recording, num_bins=30):
    """Calculate entropy of rotational velocities"""
    rotations = recording.transformation.rotation
    timestamps = recording.timestamp

    if len(rotations) < 2:
        return 0.0

    angular_velocities = []
    max_angular_velocity = 8 * np.pi  # Maximum allowed angular velocity in radians/second

    for j in range(len(rotations) - 1):
        q1 = rotations[j]
        q2 = rotations[j + 1]
        delta_time = (timestamps[j + 1] - timestamps[j]) / 1000.0

        if delta_time <= 0 or delta_time > 0.1:  # Skip invalid or long intervals
            continue

        if np.dot(q1, q2) < 0:
            q2 = -q2  # Ensure quaternions are in same hemisphere

        angle = 2 * np.arccos(np.clip(np.dot(q1, q2), -1.0, 1.0))
        angular_velocity = angle / delta_time

        if angular_velocity <= max_angular_velocity:
            angular_velocities.append(angular_velocity)

    if not angular_velocities:
        return 0.0

    # Compute histogram and entropy
    histogram, _ = np.histogram(angular_velocities, bins=num_bins, density=True)
    probabilities = histogram / np.sum(histogram)
    non_zero_probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(non_zero_probabilities * np.log(non_zero_probabilities))

    return entropy

def calculate_minimum_encapsulating_hull(recording):
    """Calculate volume of minimum convex hull"""
    translations = recording.transformation.translation

    if len(translations) < 4:  # Need at least 4 points for 3D convex hull
        return 0.0

    try:
        hull = ConvexHull(translations)
        return hull.volume
    except Exception as e:
        print(f"Error calculating convex hull: {e}")
        return 0.0

def calculate_movement_smoothness(recording):
    """Calculate movement smoothness using normalized jerk"""
    translations = recording.transformation.translation
    timestamps = recording.timestamp

    if len(translations) < 4:  # Need at least 4 points for jerk calculation
        return 0.0

    try:
        # Convert timestamps to seconds and ensure it's 1D
        time = timestamps.ravel() / 1000.0

        # Calculate velocities
        vel = np.zeros_like(translations)
        acc = np.zeros_like(translations)
        jerk = np.zeros_like(translations)

        # Calculate derivatives for each dimension separately
        for dim in range(translations.shape[1]):
            vel[:, dim] = np.gradient(translations[:, dim], time)
            acc[:, dim] = np.gradient(vel[:, dim], time)
            jerk[:, dim] = np.gradient(acc[:, dim], time)

        # Calculate normalized jerk metric
        movement_duration = time[-1] - time[0]
        path_length = np.sum(np.sqrt(np.sum(np.diff(translations, axis=0)**2, axis=1)))

        if path_length > 0 and movement_duration > 0:
            # Calculate total jerk magnitude
            jerk_magnitude = np.sqrt(np.sum(jerk**2, axis=1))
            mean_squared_jerk = np.mean(jerk_magnitude**2)

            # Normalize jerk
            normalized_jerk = np.sqrt(mean_squared_jerk) * (movement_duration**3 / path_length)
            smoothness = -np.log(normalized_jerk) if normalized_jerk > 0 else 0

            return smoothness
        return 0.0

    except Exception as e:
        print(f"Error calculating smoothness: {e}")
        return 0.0

def calculate_coordination_index(recording):
    """Calculate coordination between positional and rotational movements"""
    translations = recording.transformation.translation
    rotations = recording.transformation.rotation
    timestamps = recording.timestamp

    if len(translations) < 2 or len(rotations) < 2:
        return 0.0

    try:
        # Calculate positional and rotational velocities
        pos_vel = np.gradient(translations, timestamps/1000.0, axis=0)
        rot_vel = np.gradient(rotations, timestamps/1000.0, axis=0)

        # Calculate speeds (magnitude of velocity vectors)
        pos_speed = np.linalg.norm(pos_vel, axis=1)
        rot_speed = np.linalg.norm(rot_vel, axis=1)

        # Calculate correlation coefficient
        correlation = np.corrcoef(pos_speed, rot_speed)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except Exception as e:
        print(f"Error calculating coordination index: {e}")
        return 0.0

def calculate_spatial_efficiency(recording):
    """Calculate spatial efficiency as ratio of direct path to actual path"""
    translations = recording.transformation.translation

    if len(translations) < 2:
        return 0.0

    try:
        # Calculate total path length
        path_length = np.sum(np.sqrt(np.sum(np.diff(translations, axis=0)**2, axis=1)))

        # Calculate direct distance between start and end points
        direct_distance = np.linalg.norm(translations[-1] - translations[0])

        if path_length > 0:
            return direct_distance / path_length  # Will be between 0 and 1
        return 0.0
    except Exception as e:
        print(f"Error calculating spatial efficiency: {e}")
        return 0.0


def calculate_procedure_time(recording):
  return len(recording.timestamp)


def extract_all_features(recording):
    """Extract all features for a single recording"""
    features = []

    # Basic movement features
    path_length, avg_velocity = calculate_translational_path_and_speed(recording)
    features.extend([path_length, avg_velocity])

    # Velocity-based features
    velocity_std = calculate_standard_deviation_of_velocity(recording)
    velocity_freq_centroid = calculate_velocity_frequency_centroids(recording)
    velocity_entropy = calculate_velocity_entropy(recording)
    features.extend([velocity_std, velocity_freq_centroid, velocity_entropy])

    # Rotational features
    rot_path, rot_speed = calculate_rotational_path_and_speed(recording)
    rot_speed_std = calculate_rotational_speed_std(recording)
    rot_entropy = calculate_rotational_entropy(recording)
    rot_freq_centroid = calculate_rotational_frequency_centroids(recording)
    features.extend([rot_path, rot_speed, rot_speed_std, rot_entropy, rot_freq_centroid])

    # Advanced metrics
    smoothness = calculate_movement_smoothness(recording)
    coordination = calculate_coordination_index(recording)
    efficiency = calculate_spatial_efficiency(recording)
    features.extend([coordination, efficiency, smoothness])

    # Spatial features
    point_density = calculate_point_density_filtered(recording)
    hull_volume = calculate_minimum_encapsulating_hull(recording)
    procedure_time = calculate_procedure_time(recording)
    features.extend([point_density, hull_volume, procedure_time])

    return np.array(features)

def process_all_recordings(recordings):
    
    all_features = []
    i = 0
    for recording in recordings:
        print(f"recording: {i} / {len(recordings)}")
        i = i + 1
        features = extract_all_features(recording)
        all_features.append(features)
    return np.array(all_features)


# Extract features
features = process_all_recordings(recordings)

print(f"Extracted {features.shape[1]} features for {features.shape[0]} recordings")

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

class SurgicalSkillNet(nn.Module):
    def __init__(self, input_size=16, hidden_size=24): 
        super(SurgicalSkillNet, self).__init__()

        self.model = nn.Sequential(
            # First layer
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4), 

            # Second layer
            nn.Linear(hidden_size, hidden_size//2), 
            nn.ReLU(),
            nn.Dropout(0.4),  

            # Output layer
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def preprocess_data(features, labels):
    # Convert labels to 0/1
    labels = (labels >= 55).astype(int)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.20, random_state=42, stratify=labels)

    return X_train, X_val, y_train, y_val


def train_surgical_model(features, labels, epochs=100, batch_size=32, learning_rate=0.001, n_splits=5):


    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=42)

    # Lists to store metrics for each fold
    all_train_losses = []
    all_test_losses = []
    all_train_accuracies = []
    all_test_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(sss.split(features, labels)):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")

        X_train = features[train_idx]
        X_test = features[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        X_train = torch.tensor(X_train, dtype=torch.float64)
        y_train = torch.tensor(y_train, dtype=torch.float64)
        X_test = torch.tensor(X_test, dtype=torch.float64)
        y_test = torch.tensor(y_test, dtype=torch.float64)

        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        print(f"Training class distribution: {np.bincount(y_train.numpy().astype(int))}")
        print(f"Test class distribution: {np.bincount(y_test.numpy().astype(int))}")

        model = SurgicalSkillNet(input_size=features.shape[1])
        model = model.double()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                correct_train += (predicted == batch_y.unsqueeze(1)).sum().item()
                total_train += batch_y.size(0)

            # Calculate training metrics
            train_loss = running_loss / (len(X_train) / batch_size)
            train_accuracy = correct_train / total_train

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test.unsqueeze(1))
                predicted = (test_outputs >= 0.5).float()
                correct_test = (predicted == y_test.unsqueeze(1)).sum().item()
                test_accuracy = correct_test / len(y_test)

            # Store metrics
            train_losses.append(train_loss)
            test_losses.append(test_loss.item())
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}]:')
                print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
                print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n')

        # Store results for this fold
        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)
        all_train_accuracies.append(train_accuracies)
        all_test_accuracies.append(test_accuracies)

    # Calculate and plot average metrics across folds
    avg_train_losses = np.mean(all_train_losses, axis=0)
    avg_test_losses = np.mean(all_test_losses, axis=0)
    avg_train_accuracies = np.mean(all_train_accuracies, axis=0)
    avg_test_accuracies = np.mean(all_test_accuracies, axis=0)

    plt.figure(figsize=(12, 4))

    # Plot average losses
    plt.subplot(1, 2, 1)
    plt.plot(avg_train_losses, label='Average Train Loss')
    plt.plot(avg_test_losses, label='Average Test Loss')
    plt.title('Average Training and Test Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot average accuracies
    plt.subplot(1, 2, 2)
    plt.plot(avg_train_accuracies, label='Average Train Accuracy')
    plt.plot(avg_test_accuracies, label='Average Test Accuracy')
    plt.title('Average Training and Test Accuracy Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nFinal Average Metrics Across All Folds:")
    print(f"Train Loss: {avg_train_losses[-1]:.4f}")
    print(f"Test Loss: {avg_test_losses[-1]:.4f}")
    print(f"Train Accuracy: {avg_train_accuracies[-1]:.4f}")
    print(f"Test Accuracy: {avg_test_accuracies[-1]:.4f}")

    return model

features = np.array(features)  
labels = np.array(labels) 


X_train, X_val, y_train, y_val = preprocess_data(features, labels)

# Train the model
model = train_surgical_model(X_train, y_train, epochs=250, learning_rate=0.0002, n_splits=5)

from sklearn.metrics import classification_report

model.eval()
X_val = torch.tensor(X_val, dtype=float)
y_val = torch.tensor(y_val, dtype=float)
with torch.no_grad():
    test_outputs = model(X_val)
    predicted = (test_outputs >= 0.5).float()
    correct_test = (predicted == y_val.unsqueeze(1)).sum().item()
    validation_accuracy = correct_test / len(y_val)
    print(f"Validation Accuracy : {validation_accuracy}")

    print(classification_report(y_val, predicted, labels=[0, 1]))