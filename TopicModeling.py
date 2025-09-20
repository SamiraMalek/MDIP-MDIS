import pandas as pd
import numpy as np
from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Load the CSV file
df = pd.read_csv('/path/to/Misinfo.csv')

# Define prefixes to remove
PREFIXES_TO_REMOVE = ['MISSING CONTEXT:', 'FALSE:', 'MISLEADING:', 'NO EVIDENCE:', 'PARTLY FALSE:']

def remove_prefixes(title):
    """Remove specified prefixes from the title."""
    for prefix in PREFIXES_TO_REMOVE:
        if title.startswith(prefix):
            return title[len(prefix):].strip()
    return title

# Clean the titles
df['New_Title'] = df['Title'].apply(remove_prefixes)

# Initialize UMAP for dimensionality reduction
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=1234)

# Initialize BERTopic model
topic_model = BERTopic(umap_model=umap_model, min_topic_size=25)

# Fit the model on the titles
topics, probabilities = topic_model.fit_transform(df['Title'])

df['topic_label'] = topics  # Assign topics to dataframe

# Save topic information
topic_info = topic_model.get_topic_info()
topic_info.to_csv('/path/to/covid_topic_info.csv', index=False)

# Load sentence transformer model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode titles into embeddings
embeddings = sentence_model.encode(df['Title'].tolist(), show_progress_bar=False)

# Apply UMAP for dimensionality reduction
reduced_embeddings = umap_model.fit_transform(embeddings)

# Convert to DataFrame
embedding_df = pd.DataFrame(reduced_embeddings, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# Merge with original data
merged_df = pd.concat([df, embedding_df], axis=1)

# Compute mean values for each topic (excluding -1)
numeric_columns = ['x1', 'x2', 'x3', 'x4', 'x5']
mean_values = merged_df.groupby('topic_label')[numeric_columns].mean().drop(-1, errors='ignore')

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm1 * norm2) if norm1 and norm2 else 0

# Compute cosine-similarity for docs with label
for mean_index, mean_row in mean_values.iterrows():
    for index, row in merged_df[merged_df['topic_label'] == mean_index].iterrows():
        row_vector = row[numeric_columns].values
        similarity = cosine_similarity(row_vector, mean_row.values)
        merged_df.at[index, 'cosine_similarity'] = similarity
        
# Assign the best matching topic to documents labeled as -1
for index, row in merged_df[merged_df['topic_label'] == -1].iterrows():
    row_vector = row[numeric_columns].values
    best_label, max_similarity = None, -1
    
    for mean_index, mean_row in mean_values.iterrows():
        similarity = cosine_similarity(row_vector, mean_row.values)
        if similarity > max_similarity:
            best_label, max_similarity = mean_index, similarity
    
    merged_df.at[index, 'topic_label'] = best_label
    merged_df.at[index, 'cosine_similarity'] = max_similarity

# Save final results
merged_df.to_csv('/path/to/covid_misinfo_docs_topic_info.csv', index=False)