import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import the dataset
df = pd.read_csv("space_missions.csv", encoding='latin1')  # Use 'latin1' directly to avoid Unicode errors

# 2. Get info about the dataset
print(df.info())
print(df.head())

# 3. Handling missing data
df['Time'].fillna('00:00:00', inplace=True)

# Clean 'Price' column
df['Price'] = df['Price'].str.replace('$', '')
df['Price'] = df['Price'].str.replace(',', '')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Price'].fillna(df['Price'].mean(), inplace=True)

# Convert 'Date' column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df['Year'] = df['Date'].dt.year

# Add 'Country' column
df['Country'] = df['Location'].apply(lambda x: x.split(',')[-1].strip())

# 4. Achieve the objectives

# 1) Analyze Launch Trends Over Time
plt.figure(figsize=(10, 5))
df['Year'].value_counts().sort_index().plot(kind='line', marker='o')
plt.title("Launches Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Launches")
plt.grid()
plt.show()

# 2) Compare Success Rates by Company
plt.figure(figsize=(12, 5))
sns.countplot(data=df, x='Company', hue='MissionStatus')
plt.title("Mission Success and Failures by Company")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 3) Explore Mission Types
df['Mission_Type'] = df['Mission'].apply(lambda x: 'Communication' if 'Comm' in x else 'Other')
sns.countplot(data=df, x='Mission_Type')
plt.title("Types of Missions")
plt.show()

# 4) Cost Analysis Over Years
plt.figure(figsize=(10, 5))
df.groupby('Year')['Price'].mean().plot()
plt.title("Average Mission Cost Over Years")
plt.xlabel("Year")
plt.ylabel("Average Price")
plt.grid()
plt.show()

# 5) Geographic Launch Distribution
plt.figure(figsize=(12, 5))
df['Country'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Countries by Launch Count")
plt.xlabel("Country")
plt.ylabel("Number of Launches")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Relationships Between Variables
sns.pairplot(df[['Price', 'Year']])
plt.show()

# 6. Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[['Price', 'Year']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 7. Outliers - Boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Price'])
plt.title("Boxplot of Mission Prices")
plt.show()

# Z-score Method for Outliers (Optional Print)
z = (df['Price'] - df['Price'].mean()) / df['Price'].std()
print("Number of Price Outliers (z > 3):", (abs(z) > 3).sum())

# Extra: Simple Bar Graph - Top Launching Companies
df['Company'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title("Top 10 Launch Companies")
plt.ylabel("Number of Launches")
plt.show()
