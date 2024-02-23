import pandas as pd
from pandasgui import show
import matplotlib.pyplot as plt
import re
from sklearn.impute import SimpleImputer

#Load the data
try:
    df = pd.read_csv("C:/Users/achit/OneDrive/Desktop/data/best_selling_books.csv", error_bad_lines=False)
except AttributeError:
    # For newer versions of pandas, handle differently or inspect the file manually.
    pass

# Analyzing the data

df.head()

df.shape

df.isna().sum()


# Initial observation indicated misalignment in column names and actual data
# Correct column names based on the actual content
df.columns = ['Title', 'Author', 'Publication Year', 'Sales', 'Genre']

# Convert 'Publication Year' to numeric, setting errors to 'coerce' to handle non-numeric data
df['Publication Year'] = pd.to_numeric(df['Publication Year'], errors='coerce')

# Define a function to extract numeric sales data from the 'Sales' column, assuming sales are in millions
def extract_sales(value):
    if pd.isnull(value):
        return None
    # Extract numbers and convert them to float
    numbers = re.findall(r'\d+\.?\d*', value)
    if numbers:
        # Assume the first number is the sales figure in millions
        return float(numbers[0])
    return None

# Apply the function to extract and convert sales data
df['Sales (Millions)'] = df['Sales'].apply(extract_sales)

# Drop the original 'Sales' column as it's now redundant with 'Sales (Millions)' containing the cleaned data
df.drop('Sales', axis=1, inplace=True)


# Creating a SimpleImputer object for numeric data
numeric_imputer = SimpleImputer(strategy='median')

# Columns to impute
numeric_cols = ['Publication Year', 'Sales (Millions)']

# Applying imputer to the numeric columns
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')

# Reshape the 'Genre' column since SimpleImputer expects 2D data
genres = df['Genre'].values.reshape(-1, 1)

# Applying the imputer to the 'Genre' column
df['Genre'] = categorical_imputer.fit_transform(genres)

# Check to ensure no missing values remain in 'Genre'
print(df['Genre'].isnull().sum())


# Display the cleaned dataframe's information and the first few rows to confirm the cleaning process
print(df.info())
print(df.head())


df.isna().sum()

df['Publication Year'].dropna().plot(kind='hist', bins=20)
plt.title('Distribution of Publication Years')
plt.xlabel('Publication Year')
plt.ylabel('Frequency')
plt.show()


# Group the data by 'Genre' and calculate the mean sales for each genre
genre_sales = df.groupby('Genre')['Sales (Millions)'].mean().sort_values(ascending=False)

# Plotting
plt.figure(figsize=(10, 8))  # Adjust the figure size as necessary
genre_sales.plot(kind='bar')
plt.title('Average Sales by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Sales (Millions)')
plt.xticks(rotation=45, ha='right')  # Rotate the genre names for better readability
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

# Show the plot
plt.show()