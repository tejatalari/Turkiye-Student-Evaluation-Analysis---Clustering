

# Turkiye Student Evaluation Analysis - Clustering

This repository contains a project that focuses on the analysis of student evaluations using clustering techniques. The data, collected from university students in Turkey, contains evaluations of instructors and courses, which are then clustered to uncover patterns and insights about student perceptions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Clustering Techniques](#clustering-techniques)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project involves the application of clustering algorithms to a dataset containing student evaluations of instructors and courses. The goal is to group similar evaluations together and identify patterns that could provide insights into student satisfaction and educational quality.

## Dataset

The dataset used in this project is the "Turkiye Student Evaluation" dataset. It contains evaluations from university students in Turkey, focusing on various aspects of courses and instructors. The dataset includes the following columns:

- `instructor_id`: Unique identifier for each instructor.
- `course_id`: Unique identifier for each course.
- `evaluation`: Student evaluation score (1-5).
- `question_1` to `question_28`: Evaluation responses to 28 different questions related to the course and instructor.

The dataset is available on the UCI Machine Learning Repository.

## Installation

To run this project, you'll need to have Python installed along with the necessary libraries. You can install the required dependencies using pip:

```bash
git clone https://github.com/your-username/turkiye-student-evaluation-clustering.git
cd turkiye-student-evaluation-clustering
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Data Preparation

The dataset is loaded and preprocessed to prepare it for clustering:

1. **Loading Data**: The data is loaded using `pandas`.
2. **Handling Missing Values**: Any missing values are handled appropriately.
3. **Feature Scaling**: The data is scaled to ensure that all features contribute equally to the clustering process.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('turkiye-student-evaluation.csv')

# Handle missing values (if any)
df.fillna(method='ffill', inplace=True)

# Feature scaling
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df.iloc[:, 3:])
```

## Clustering Techniques

In this project, various clustering algorithms are applied to the dataset to identify distinct groups of student evaluations. The following techniques are used:

- **K-Means Clustering**: A popular clustering algorithm that partitions the data into k clusters, where each data point belongs to the cluster with the nearest mean.
- **Hierarchical Clustering**: A method that builds a hierarchy of clusters, often visualized using a dendrogram.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: A clustering algorithm that groups together points that are closely packed together while marking points that lie alone in low-density regions as outliers.

```python
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_df)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(scaled_df)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_df)
```

## Evaluation

The performance of the clustering models is evaluated using metrics such as the silhouette score, which measures how similar a point is to its own cluster compared to other clusters. Additionally, visual inspection of the clusters is done using dimensionality reduction techniques like PCA.

```python
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Silhouette Score for K-Means
kmeans_silhouette = silhouette_score(scaled_df, kmeans_labels)

# PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_df)

# Visualizing K-Means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=kmeans_labels, palette='viridis')
plt.title('K-Means Clustering')
plt.show()
```

## Visualization

The clusters are visualized using scatter plots, which help in understanding the grouping of the data points. This visualization provides insights into the different clusters formed by each algorithm and their characteristics.

## Usage

To run the analysis, execute the Python script provided in this repository. The script will load the data, perform clustering, and display the results, including visualizations and evaluation metrics.

```bash
python turkiye_student_clustering.py
```

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


