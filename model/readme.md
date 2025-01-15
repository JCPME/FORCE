
# Time–Frequency Analysis with TCN and FFN for Binary Classification

This script contains a complete pipeline for training a Temporal Convolutional Network (TCN) with an additional feedforward network (FFN) on time–frequency features extracted via the Short-Time Fourier Transform (STFT) from kinematic surgery tool and bone data. 

## Overview

The code performs the following:

1. **Data Loading and Preparation:**
   - Loads a pickled dataset containing participant motion data.
   - Creates a binary label based on whether a participant’s average GRS score is above or below the mean.
   - Filters data based on specified tools and splits the data into training, validation, and test sets using participant IDs.
  
2. **Data Augmentation and Preprocessing:**
   - Combines translation and rotation arrays for each sample into a single array.
   - Augments the kinematic data using operations such as spatial rotation, mirroring, and the addition of Gaussian noise.
   - Extracts 2D time–frequency features using the STFT.
   - Scales the extracted features and additional numeric features (e.g., tool, case, total procedure time).

3. **Dataset and Dataloader Creation:**
   - Constructs PyTorch datasets and dataloaders, where the STFT outputs and additional features are organized for training.

4. **Model Definition:**
   - Defines custom network components including temporal blocks with causal convolutions and optional self-attention.
   - Builds a hybrid TCN-FFN architecture where the TCN processes the STFT feature maps and the FFN processes additional numerical features.
   - Implements a classifier layer on top of the concatenated TCN and FFN outputs.

5. **Training and Evaluation:**
   - Uses a weighted binary cross-entropy loss to account for class imbalances.
   - Trains the model with early stopping based on validation performance.
   - Evaluates the final model on a test set, producing classification metrics, ROC curves, and confusion matrices.
   - Visualizes training progress (loss and accuracy over epochs).


## Data

- **Input Data:**  
  The code expects a pickled pandas DataFrame (`dataset.pkl`) located under the `../data/` directory (relative to the script). Make sure this file is accessible and structured with the following columns (among others):

  - `participant_num`
  - `translation_array` – Array of translation data
  - `rotation_array` – Array of rotation (quaternion) data
  - `timestamp_array`
  - `avg_grs_score`
  - `total_procedure_time`
  - `tool`
  - `case`

- **Label Creation:**  
  A binary label is generated based on whether the average GRS score is above or below the global mean.

## Usage Model

After installing the dependencies and preparing your data, you can run the main script. The pipeline includes data loading, augmentation, feature extraction, model training, and evaluation.
Then run this file from the root folder.

```bash
python .\model\model.py
```

If you are using a Jupyter Notebook, ensure that the notebook’s kernel has access to the required packages and that your dataset path is correct.

## Usage jigsaws.py 
Upload the the jigsawsdataset.pkl into the data folder, then run this file from the root folder.
```bash
python .\model\jigsaws.py
```

## Code Structure

- **Installation Instructions:**  
  A comment at the top of the code file provides installation instructions.

- **Data Preprocessing and Augmentation:**  
  Functions are defined to combine translation and rotation arrays, augment data (via rotation, mirroring, and noise), and generate STFT features.

- **Feature Extraction:**  
  Feature matrices are built for training, validation, and testing. Scaling of both STFT features and additional features is applied.

- **Dataset Classes and DataLoaders:**  
  Custom PyTorch datasets (`FFTAdditionalDataset`) are created to serve data batches to the model.

- **Model Components:**  
  - `Chomp1d` and `TemporalBlock`: Custom layers ensuring causal convolutions.
  - `TCN`: Stacks multiple temporal blocks.
  - `TCN_FFN_Model`: A hybrid model combining the TCN with a feedforward network for additional features and a classifier.

- **Training and Evaluation:**  
  The model is trained using the Adam optimizer with an early stopping mechanism based on validation loss. Classification metrics and ROC curves are generated to evaluate performance on the test set.




