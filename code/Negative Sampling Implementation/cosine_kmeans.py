#%%
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
import random
from sklearn.cluster import KMeans
import re
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Initialize the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Combine triples as a list
raw_dir = '/Users/yoninayoni/Documents/GitHub/Capstone/data/MetaQA/raw'

for split in ['train', 'valid', 'test']:
    with open(os.path.join(raw_dir, f'{split}.txt'), 'r') as f:
        # Read the file and split into lines, then split each line into components to form triples
        lines = f.read().split('\n')[:-1]
        triples = [row.split('\t') for row in lines]  # Each 'row' is split into head, relation, tail
        
        # Extract relations from triples
        relations = [triple[1] for triple in triples]


# 4. Convert triples into sentences
triple_sentences = [' '.join(triple) for triple in triples]

# 5. Convert triples and relations to embeddings
triple_embeddings = model.encode(triple_sentences) # (4053, 384)
relation_embeddings = model.encode(relations) # (4053, 384)


#%%
# 6. Sample question

# Path to the text file
question_path = '/Users/yoninayoni/Documents/GitHub/Capstone/data/MetaQA/qa_dev_1hop.txt'

# Initialize an empty list to store the questions
questions = []

# Open and read the file
with open(question_path, 'r') as file:
    for line in file:
        # Split the line at the tab character
        parts = line.strip().split('\t')
        if len(parts) > 1:
            question = parts[0]
            # Remove the brackets and anything inside them
            cleaned_question = question.replace('[', '').replace(']', '')
            # Append the cleaned question to the list
            questions.append(cleaned_question)


# Print or process the list of questions
print(questions)

#%%
unique_relations_set = set(relations)

# # Convert the set back to a list if you need list operations later
# unique_relations_list = list(unique_relations_set)

# Print the unique relations
print("Unique relations:", unique_relations_set)
print("Number of unique relations:", len(unique_relations_set))

relation_kmeans = KMeans(n_clusters=9, random_state=42)
relation_fit = relation_kmeans.fit(relation_embeddings)
relation_clusters = relation_kmeans.fit_predict(relation_embeddings)

# Map each relation to its cluster
relation_to_cluster = {relation: cluster for relation, cluster in zip(relations, relation_clusters)}

#%%
results = []

for i, question in enumerate(questions):
    # Convert question to embeddings
    question_embedding = model.encode([question])

    # Calculate cosine similarities
    similarities = cosine_similarity(question_embedding, triple_embeddings)
    relation_similarities = cosine_similarity(question_embedding, relation_embeddings)

    # Find the index of the highest similarity score
    most_similar_index = np.argmax(similarities)
    most_similar_score = similarities[0, most_similar_index]
    
    # Find the most similar relation
    most_similar_relation_index = np.argmax(relation_similarities)
    most_similar_relation = relations[most_similar_index]
    
    # Find the cluster of the most similar relation
    question_topic_cluster = relation_to_cluster[most_similar_relation]
    
    # Append the results
    results.append({
        "Question": question,
        "Question Embedding": question_embedding,
        "Most Similar Triple": ' '.join(triples[most_similar_index]),
        # "Answer": triples[most_similar_index][2],
        "Cosine Similarity Score": most_similar_score,
        "Most Similar Relation": most_similar_relation,
        "Topic Cluster": question_topic_cluster
    })

# Convert results to a pandas DataFrame
df_results = pd.DataFrame(results)

# Set the threshold for high similarity
threshold = 0.7

# Add a new column 'High Similarity' to the DataFrame
# It will contain 1 if the cosine similarity score is above 0.8, and 0 otherwise
df_results['High Similarity'] = (df_results['Cosine Similarity Score'] >= threshold).astype(int)

#%% Compare Results

# Path to the text file
modeling_question_path = '/Users/yoninayoni/Downloads/ComplEx_RoBERTa_best_score_model (1).txt'

# Read the file into a DataFrame
df = pd.read_csv(modeling_question_path, sep='\t', header=None, names=['Question', 'ID', 'Flag'])

# Counting '1's in the Flag column of df
modeling_count = df['Flag'].sum()

# Counting '1's in the High Similarity column of df_results
high_similarity_count = df_results['High Similarity'].sum()

# Printing the counts
print("Number of '1's in KGQA:", modeling_count)
print("Number of '1's in Classical Method:", high_similarity_count)

# Determining which has more '1's
if modeling_count > high_similarity_count:
    print("KGQA has more '1's.")
elif modeling_count < high_similarity_count:
    print("Cosine Similarity has more '1's.")
else:
    print("Both columns have the same number of '1's.")


#=====================================================================
#%%1 once
# Find topic of questions
qe_array = np.vstack(df_results["Question Embedding"])
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


# Step 2: Perform hierarchical clustering
model = AgglomerativeClustering(distance_threshold=None, n_clusters=9, linkage='ward')
clusters = model.fit_predict(qe_array)

# Step 3: Visualize the dendrogram (Optional)
# This requires a full linkage matrix which can be computed from the embeddings
linked = linkage(qe_array, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()




#%%2 once for 9 
# Assuming df_results is already loaded and contains the columns needed
linkages = {}
labels_dict = {}

for relation in sorted(unique_relations_set):
    group_df = df_results[df_results['Most Similar Relation'] == relation]
    embeddings = np.vstack(group_df['Question Embedding'])
    labels = group_df['Question'].tolist()  # Get the corresponding questions as labels
    
    # Perform hierarchical clustering
    linkages[relation] = linkage(embeddings, method='ward')
    labels_dict[relation] = labels  # Store labels in a dictionary

# Visualize All Clusters in a Unified Dendrogram
fig, axes = plt.subplots(nrows=len(unique_relations_set), ncols=1, figsize=(10, 20))

for ax, relation in zip(axes, sorted(unique_relations_set)):
    dendrogram(linkages[relation], ax=ax, orientation='top', labels=labels_dict[relation], distance_sort='descending', show_leaf_counts=True)
    ax.set_title(f'Cluster: {relation}')

plt.tight_layout()
plt.show()


#%% once for 3 

# Example to split visualizations
num_relations = len(unique_relations_set)
relations_per_figure = 3  # Number of dendrograms per figure

for i in range(0, num_relations, relations_per_figure):
    fig, axes = plt.subplots(nrows=relations_per_figure, ncols=1, figsize=(10, 15))
    for ax, relation in zip(axes, sorted(unique_relations_set)[i:i+relations_per_figure]):
        dendrogram(linkages[relation], ax=ax, orientation='top', labels=labels_dict[relation], distance_sort='descending', show_leaf_counts=True)
        ax.set_title(f'Cluster: {relation}')
    plt.tight_layout()
    plt.show()
    
    
#%%
# Find topic of questions
qe_array = np.vstack(df_results["Question Embedding"])
kmeans_ = KMeans(n_clusters=5, random_state=42).fit(qe_array)
question_topic = kmeans_.labels_

df_results['Question Topic'] = question_topic


# Plot questions
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(qe_array)

# #%% simple plot
# # Plotting the 2D embeddings
# plt.figure(figsize=(12, 8))

# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue')
# for i, question in enumerate(questions):
#     plt.annotate(question, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
# plt.xlabel('Component 1')
# plt.ylabel('Component 2')
# plt.title('2D PCA of Question Embeddings')
# plt.show()

colors = np.array(['red', 'green', 'blue', 'purple', 'orange'])

plt.figure(figsize=(12, 8))

# Here, we use the 'c' argument in scatter to assign colors based on 'question_topic'
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors[question_topic])

for i, question in enumerate(questions):
    plt.annotate(question, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9, alpha=0.7)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('2D PCA of Question Embeddings with Topic Grouping')
plt.show()

#%% second
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
# Assuming df_results already has 'Question Embedding' and 'Question Topic' as columns

# Stack the question embeddings for PCA
qe_array = np.vstack(df_results["Question Embedding"])

# Apply PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(qe_array)

# Colors for the clusters - make sure you have enough colors for all clusters
colors = np.array(['red', 'green', 'blue', 'purple', 'orange', 'yellow'])

unique_clusters = np.unique(df_results['Topic Cluster'])

if len(unique_clusters) > len(colors):
    raise ValueError("Not enough colors for the number of clusters.")

# Creating the plot
plt.figure(figsize=(12, 8))

# Map each unique cluster ID to a color and a label
cluster_color_map = {cluster: color for cluster, color in zip(unique_clusters, colors)}
cluster_label_map = {cluster: relation for cluster, relation in zip(unique_clusters, df_results['Most Similar Relation'].unique())}


# Scatter plot with color mapping by 'Topic Cluster'
scatter_colors = [cluster_color_map[cluster] for cluster in df_results['Topic Cluster']]
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=scatter_colors, alpha=0.7)

# Create a custom legend
legend_handles = [Patch(color=cluster_color_map[cluster], label=cluster_label_map[cluster]) for cluster in unique_clusters]
plt.legend(handles=legend_handles, title="Relation Topics")

#
# # Scatter plot with annotations
# scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors[df_results['Topic Cluster']], alpha=0.7)
# plt.colorbar(scatter, ticks=np.unique(df_results['Topic Cluster']))
# plt.clim(-0.5, len(np.unique(df_results['Topic Cluster'])) - 0.5)

# Add annotations for each question
for i, question in enumerate(df_results['Question']):
    plt.annotate(question, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9, alpha=0.7)

# Adding labels and title
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D PCA of Question Embeddings with Topic Grouping')

# Display the plot
plt.show()


#%%
# Display or save the DataFrame
df_results = df_results.drop(['Question Embedding'], axis=1)
# To save the DataFrame as a CSV file
# df_results.to_csv('qa_results.csv', index=True)


# %%
# # Flatten the array to make it a 1D array
# flattened_similarities = similarities.flatten()

# # Use argsort() which returns the indices that would sort the array
# sorted_indices = np.argsort(flattened_similarities)

# # The last element is the index of the highest value, so the second last is what you're looking for
# second_highest_index = sorted_indices[-2]

# # Get the second highest similarity score
# second_highest_similarity = flattened_similarities[second_highest_index]

# # Output the result
# print(f"Second highest cosine similarity score: {second_highest_similarity:.4f}")
# print(f"Corresponding triple: {triples[second_highest_index]}")


#%%
# Make 20 questions and set the threshord (0.8). Compare the result with Sen&Nina's result
# To check topic of question using clustering (K means). Combine question embedding/ answering embedding.




# # %%
# hierarch clustering
# for example,
# cluster1 has 100 questions and 100 questions have each relations. Sort 100 questions