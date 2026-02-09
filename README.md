### EX4 Implementation of Cluster and Visitor Segmentation for Navigation patterns

### AIM: To implement Cluster and Visitor Segmentation for Navigation patterns in Python.
### Description:
<div align= "justify">Cluster visitor segmentation refers to the process of grouping or categorizing visitors to a website, 
  application, or physical location into distinct clusters or segments based on various characteristics or behaviors they exhibit. 
  This segmentation allows businesses or organizations to better understand their audience and tailor their strategies, marketing efforts, 
  or services to meet the specific needs and preferences of each cluster.</div>
  
### Procedure:
1) Read the CSV file: Use pd.read_csv to load the CSV file into a pandas DataFrame.
2) Define Age Groups by creating a dictionary containing age group conditions using Boolean conditions.
3) Segment Visitors by iterating through the dictionary and filter the visitors into respective age groups.
4) Visualize the result using matplotlib.

### Program 1:
```
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv(r"C:\Users\admin\Downloads\clustervisitor.csv")

# Define age-based clusters
cluster = {
    "Young": (df['Age'] <= 30),
    "Middle": ((df['Age'] > 30) & (df['Age'] <= 45)),
    "Old": (df['Age'] > 45)
}

# Lists for plotting
age_group_labels = []
visitor_counts = []

# Count visitors in each group
for group, condition in cluster.items():
    visitors = df[condition]
    visitors_count = len(visitors)

    age_group_labels.append(group)
    visitor_counts.append(visitors_count)

    print(f"\nVisitors in {group} age group")
    print(visitors.to_string(index=False))
    print("Count of Visitors:", visitors_count)

# Plot age-based distribution
plt.figure(figsize=(8, 6))
plt.bar(age_group_labels, visitor_counts)
plt.xlabel('Age Groups')
plt.ylabel('Number of Visitors')
plt.title('Visitor Distribution Across Age Groups')
plt.show()
```
### Output:

<img width="627" height="799" alt="image" src="https://github.com/user-attachments/assets/780f77ff-054b-4a59-84ea-bc63ce38880d" />

### Program 2:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Clean column names (Excel safety)
df.columns = df.columns.str.strip()

# Remove existing Cluster column if present
if 'Cluster' in df.columns:
    df = df.drop(columns=['Cluster'])

# Select features for clustering
X = df[['Age', 'Income']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Convert cluster labels to DataFrame
cluster_df = pd.DataFrame(clusters, columns=['Cluster'])

# Concatenate cluster labels with original dataset
df = pd.concat([df, cluster_df], axis=1)

# -----------------------------------
# Arrange clusters by average Income
# -----------------------------------
cluster_order = (
    df.groupby('Cluster')['Income']
    .mean()
    .sort_values()
    .index
)

cluster_mapping = {old: new for new, old in enumerate(cluster_order)}
df['Cluster'] = df['Cluster'].map(cluster_mapping)

# -----------------------------------
# Display clustering result neatly
# -----------------------------------
print("\nK-Means Clustering Result (Ordered by Income)")
print(df[['Age', 'Income', 'Cluster']].sort_values('Cluster').to_string(index=False))

# -----------------------------------
# Visualize clusters
# -----------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['Income'], c=df['Cluster'])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("K-Means Clustering using Age and Income (Low → High)")
plt.show()

```
### Output:

<img width="630" height="702" alt="image" src="https://github.com/user-attachments/assets/c6d23b33-a540-4888-ae12-f0e09d440490" />


### Result:

Thus the Implementation of Cluster and Visitor Segmentation for Navigation patterns is executed successfully.
