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


# 3. Convert triples into sentences
triple_sentences = [' '.join(triple) for triple in triples]

# 4. Convert triples and relations to embeddings
triple_embeddings = model.encode(triple_sentences) # (4053, 384)
relation_embeddings = model.encode(relations) # (4053, 384)


#%%
# 5. Convert questions to embeddings

# Path to the text file
question_path = '/Users/yoninayoni/Documents/GitHub/Capstone/data/MetaQA/qa_dev_1hop.txt'

# Initialize an empty list to store the questions
questions = [] # 9992

# Initialize an empty dictionary to store the mapping of questions to correct answers
correct_answers = {}

# Open and read the file
with open(question_path, 'r') as file:
    for line in file:
        # Split the line at the tab character
        parts = line.strip().split('\t')
        if len(parts) > 1:
            question, answers = parts[0], parts[1]
            # Remove the brackets and anything inside them
            cleaned_question = question.replace('[', '').replace(']', '')
            # Split the answers string into a list of answers
            answer_list = answers.split('|')
            # Append the cleaned question to the list
            questions.append(cleaned_question)
            # Add to the dictionary
            correct_answers[cleaned_question] = answer_list

# Print or process the list of questions
print(questions)


#%% 
# 6. Clustering relations
unique_relations_set = set(relations)

# Print the unique relations
print("Unique relations:", unique_relations_set)
print("Number of unique relations:", len(unique_relations_set))

relation_kmeans = KMeans(n_clusters=9, random_state=42)
relation_fit = relation_kmeans.fit(relation_embeddings)
relation_clusters = relation_kmeans.fit_predict(relation_embeddings)

# Map each relation to its cluster
relation_to_cluster = {relation: cluster for relation, cluster in zip(relations, relation_clusters)}

#%%
# 7. Convert cs results to a pandas DataFrame to see at a glance
results = []

for i, question in enumerate(questions):
    # Convert question to embeddings 
    question_embedding = model.encode([question]) # (9992, 384)

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

df_results = pd.DataFrame(results)

# Set the threshold for high similarity
threshold = 0.7

# Add a new column 'High Similarity' to the DataFrame
# It will contain 1 if the cosine similarity score is above 0.8, and 0 otherwise
df_results['High Similarity'] = (df_results['Cosine Similarity Score'] >= threshold).astype(int)

# Add the correct answers as a new column in df_results DataFrame
df_results['Correct Answer'] = df_results['Question'].map(correct_answers)


#%% 
# 8. hierarcical clustering
linkages = {}
labels_dict = {}

for relation in sorted(unique_relations_set):
    group_df = df_results[df_results['Most Similar Relation'] == relation]
    embeddings = np.vstack(group_df['Question Embedding'])
    labels = group_df['Question'].tolist()  # Get the corresponding questions as labels
    
    # Perform hierarchical clustering
    linkages[relation] = linkage(embeddings, method='ward')
    labels_dict[relation] = labels  # Store labels in a dictionary

# Example to split visualizations
num_relations = len(unique_relations_set)
relations_per_figure = 3  # Number of dendrograms per figure

for i in range(0, num_relations, relations_per_figure):
    fig, axes = plt.subplots(nrows=relations_per_figure, ncols=1, figsize=(10, 15))
    for ax, relation in zip(axes, sorted(unique_relations_set)[i:i+relations_per_figure]):
        dendrogram(linkages[relation], ax=ax, orientation='top', distance_sort='descending', show_leaf_counts=True)
        ax.set_title(f'Cluster: {relation}')
        ax.set_xticklabels([])
    plt.tight_layout()
    plt.show()
    
# ax.set_xticklabels([])
#%% 
# 9. Compare Results with cs and kgqa

# Path to the text file
modeling_question_path = '/Users/yoninayoni/Documents/GitHub/Capstone/data/MetaQA/kgqa_result.txt'

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


# #%%

# from collections import Counter
# import re

# # Join all questions into a single string
# all_questions = ' '.join(questions)

# # Tokenize the string into words using regular expression to handle punctuation
# words = re.findall(r'\w+', all_questions.lower())  # This converts all text to lowercase and finds words

# # Count the occurrences of each word
# word_count = Counter(words)

# # Display the most common words and their counts
# print(word_count.most_common())  # This prints all words sorted by frequency

# # Want to see specific words count
# print("Frequency of 'movies':", word_count['movies'])
# print("Frequency of 'films':", word_count['films'])
# print("Frequency of 'act':", word_count['act'])
# #%%
# import pandas as pd
# import re

# # Define the dictionary mapping keywords to relations
# relation_keywords = {
#     'directed_by': ['director', 'directed', 'creator', 'direct'],
#     'has_genre': ['genre', 'type', 'sort', 'kind', 'applicable'],
#     'has_imdb_rating': ['rate', 'good', 'rating', 'think', 'opinion', 'popular', 'popularity'],
#     'has_imdb_votes': ['famous'],
#     'has_tags': ['words', 'describe', 'topics', 'described', 'about'],
#     'in_language': ['language'],
#     'release_year': ['released', 'year', 'release', 'when', 'date'],
#     'starred_actors': ['starred', 'acted', 'actor', 'stars', 'act', 'star', 'actors', 'appear', 'appears'],
#     'written_by': ['writer', 'wrote', 'write', 'written', 'screenwriter', 'author']
# }

# def identify_relation(question):
#     words = re.findall(r'\w+', question.lower())  # Tokenize the question into words
#     for relation, keywords in relation_keywords.items():
#         if any(word in words for word in keywords):
#             return relation
#     return None  # Return None if no keywords match

# # Assuming 'df_results' is already DataFrame and it has a column named 'Question'
# df_results['Real Relation'] = df_results['Question'].apply(identify_relation)
