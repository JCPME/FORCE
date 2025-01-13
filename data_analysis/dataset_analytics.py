import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_path = "dataset.pkl"
df = pd.read_pickle(data_path)

# Create the label column
df['label'] = df['avg_grs_score'].apply(lambda x: 0 if x < 55 else 1)

# Check the distribution of labels
label_counts = df['label'].value_counts()
print("Label Distribution:")
print(label_counts)

procedure_time_stats = df['total_procedure_time'].describe()
print("\nProcedure Time Distribution (descriptive stats):")
print(procedure_time_stats)

# Group by 'label' and compute mean and std of 'total_procedure_time'
stats_by_label = df.groupby('label')['total_procedure_time'].agg(['mean', 'std'])

print(stats_by_label)

sns.boxplot(x='label', y='total_procedure_time', data=df)

plt.title('Procedure Time by Label')
plt.xlabel('Label (0 or 1)')
plt.ylabel('Procedure Time')
plt.show()

case_distribution = df['case'].value_counts()
print("\nCase Distribution:")
print(case_distribution)

# Plot 1: Overall GRS Distribution
plt.subplot(2, 1, 1)
sns.histplot(data=df, x='avg_grs_score', bins=20, color='skyblue')
plt.title('Overall Distribution of GRS Scores for Bone Procedures')
plt.xlabel('GRS Score')
plt.ylabel('Count')

# Add mean line
mean_grs = df['avg_grs_score'].mean()
plt.axvline(mean_grs, color='red', linestyle='--', label=f'Mean: {mean_grs:.2f}')
plt.legend()