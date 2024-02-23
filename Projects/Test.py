import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:/Users/achit/OneDrive/Desktop/data/heart_failure_clinical_records_dataset.csv")

# Descriptive Statistics
print("Descriptive Statistics:")
print(data.describe())

# Check for Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# EDA: Distribution of Key Variables
sns.set_style("whitegrid")
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Age Distribution
sns.histplot(data['age'], kde=True, ax=axs[0, 0], color='skyblue')
axs[0, 0].set_title('Age Distribution')

# Creatinine Phosphokinase Distribution
sns.histplot(data['creatinine_phosphokinase'], kde=True, ax=axs[0, 1], color='lightgreen')
axs[0, 1].set_title('Creatinine Phosphokinase Distribution')

# Ejection Fraction Distribution
sns.histplot(data['ejection_fraction'], kde=True, ax=axs[1, 0], color='salmon')
axs[1, 0].set_title('Ejection Fraction Distribution')

# Platelets Distribution
sns.histplot(data['platelets'], kde=True, ax=axs[1, 1], color='gold')
axs[1, 1].set_title('Platelets Distribution')

# Serum Creatinine Distribution
sns.histplot(data['serum_creatinine'], kde=True, ax=axs[2, 0], color='lightblue')
axs[2, 0].set_title('Serum Creatinine Distribution')

# Serum Sodium Distribution
sns.histplot(data['serum_sodium'], kde=True, ax=axs[2, 1], color='thistle')
axs[2, 1].set_title('Serum Sodium Distribution')

plt.tight_layout()
plt.show()

# Correlation Analysis
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


plt.figure(figsize=(10, 10))
sns.scatterplot(data=data, x='age', y='ejection_fraction', hue='DEATH_EVENT', style='DEATH_EVENT', palette='viridis')
plt.title('Age vs. Ejection Fraction by Death Event')
plt.xlabel('Age')
plt.ylabel('Ejection Fraction')
plt.legend(title='Death Event', labels=['Survived', 'Died'])
plt.grid(True)
plt.show()