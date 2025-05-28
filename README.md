# Word Embeddings, Clustering, and Sentiment Analysis

This project explores unsupervised learning techniques using pre-trained word embeddings and investigates their application to enhance sentiment classification. The project is divided into two main parts: word embedding exploration and clustering analysis, followed by an optional bonus section on improving sentiment classification.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Data](#data)
- [Part 1: Word Embeddings Exploration](#part-1-word-embeddings-exploration)
- [Part 2: Sentiment Classification Enhancement (Bonus)](#part-2-sentiment-classification-enhancement-bonus)
- [Results](#results)
- [Usage](#usage)
- [Key Findings](#key-findings)

## Overview

This assignment demonstrates the practical application of word embeddings in natural language processing tasks. We utilize GloVe (Global Vectors for Word Representation) embeddings to:

1. Create semantic word clusters based on similarity
2. Visualize high-dimensional word embeddings in 2D space
3. Apply clustering algorithms to discover word groupings
4. Enhance traditional bag-of-words representations for sentiment analysis

## Dependencies

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
```

## Data

The project uses:
- **GloVe_Embedder_data.txt**: A reduced subset of GloVe embeddings containing the intersection of the IA3 sentiment dataset vocabulary and the full GloVe vocabulary
- **IA3-train.csv**: Training data for sentiment classification
- **IA3-dev.csv**: Validation data for sentiment classification

## Part 1: Word Embeddings Exploration

### 1.1 Dataset Construction (10 pts)

We begin with five seed words: `'flight'`, `'awesome'`, `'terrible'`, `'help'`, and `'late'`. For each seed word, we find the 30 most similar words using Euclidean distance in the embedding space, creating a dataset of 150 words naturally grouped into 5 clusters.

**Sample Results:**
- **flight**: plane, flights, boarding, airline, jet, flying
- **awesome**: amazing, great, fantastic, cool, fun, epic
- **terrible**: horrible, awful, bad, brutal, idea, horrendous
- **help**: need, helping, please, pls, let, us
- **late**: early, earlier, usual, after, again, saturday

### 1.2 Visualization (35 pts)

#### 1.2.1 Principal Component Analysis (PCA) (15 pts)

We apply PCA to reduce the high-dimensional word embeddings to 2D for visualization. The scatter plot uses distinct colors for each seed word cluster and annotates each point with its corresponding word.

**Findings**: The PCA visualization shows five distinct clusters with reasonable separation, though some overlap exists between clusters. The clustering demonstrates that semantically similar words are positioned close to each other in the embedding space.

#### 1.2.2 t-SNE Visualization (20 pts)

We apply t-SNE with different perplexity values (5, 10, 20, 30, 40, 50) to create nonlinear 2D embeddings.

**Key Observations**:
- Lower perplexity values (5, 10) create tighter, more compact clusters but may miss broader structure
- Higher perplexity values (40, 50) emphasize global relationships with clearer inter-cluster separation
- Moderate values (20, 30) provide the best trade-off between local detail and global structure
- Excessive perplexity can lead to over-spread clusters with reduced interpretability

### 1.3 Clustering Analysis (35 pts)

#### 1.3.1 K-means Objective Function (15 pts)

We apply K-means clustering with k values ranging from 2 to 20 and plot the inertia (within-cluster sum of squares) as a function of k.

**Results**: The objective function decreases monotonically as k increases, which is expected since more clusters allow better fitting of the data points. However, no clear "elbow" indicates k=5 as optimal, suggesting the relationship between the number of clusters and data structure is more complex than anticipated.

#### 1.3.2 Evaluation Metrics (20 pts)

Using the original seed words as ground truth labels, we evaluate clustering performance using:

- **Purity**: Custom implementation measuring the fraction of correctly assigned points
- **Adjusted Rand Index (ARI)**: Measures similarity between two clusterings
- **Normalized Mutual Information (NMI)**: Measures mutual dependence between clusterings

**Key Findings**:
1. k=5 does not necessarily provide the best scores across all metrics
2. Around k=7 shows better performance for ARI and NMI
3. Purity increases with k due to overfitting (more clusters = higher purity)
4. For comparing algorithms with different cluster numbers, ARI and NMI are more appropriate than Purity

## Part 2: Sentiment Classification Enhancement (Bonus)

### Bag-of-Word-Clusters Approach

We implement an innovative approach to improve upon traditional bag-of-words representation:

1. **Clustering**: Apply K-means (k=100) to word embeddings to group semantically similar words
2. **Representation**: Transform documents from word-based to cluster-based representation
3. **Classification**: Train SVM classifier on the new representation

**Methodology**:
- Map each word to its assigned cluster
- Transform documents into "Bag-of-Word-Clusters" representation
- This addresses the issue where semantically similar words (e.g., "good" vs "pleasant") are treated as distinct in traditional BoW

**Results**: 
- Training Accuracy: 84.83%
- This approach reduces dimensionality and captures semantic relationships between words

## Key Findings

1. **Embedding Quality**: GloVe embeddings effectively capture semantic relationships, as evidenced by meaningful word clusters
2. **Visualization**: Both PCA and t-SNE reveal distinct semantic clusters, with t-SNE providing more nuanced separation
3. **Clustering**: Traditional metrics suggest optimal cluster numbers may differ from expected ground truth
4. **Classification**: Semantic clustering of words can provide an alternative to high-dimensional BoW representations

## Usage

1. **Setup**: Ensure all required libraries are installed and data files are in the correct Google Drive directory
2. **Data Loading**: Mount Google Drive and load the GloVe embeddings
3. **Part 1**: Run the word similarity analysis, visualization, and clustering sections
4. **Part 2**: Execute the sentiment classification enhancement (bonus section)

## File Structure

```
├── GloVe_Embedder_data.txt     # Pre-trained GloVe embeddings (reduced set)
├── IA3-train.csv               # Training data for sentiment analysis
├── IA3-dev.csv                 # Validation data for sentiment analysis
└── notebook.ipynb             # Main analysis notebook
```

## Technical Notes

- The `GloVe_Embedder` class provides utilities for loading embeddings, finding nearest neighbors, and handling unknown words
- All visualizations use consistent color coding for seed word clusters
- The clustering evaluation compares multiple metrics to provide comprehensive assessment
- The bonus section demonstrates practical application of embeddings to improve NLP tasks

This project demonstrates the power of word embeddings in capturing semantic relationships and their practical applications in both unsupervised learning tasks and supervised classification problems.
