#%%
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
import random

# 1. Initialize the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Combine triples as a list
raw_dir = '/Users/yoninayoni/Documents/GitHub/Capstone/data/MetaQA/raw'

for split in ['train', 'valid', 'test']:
    with open(os.path.join(raw_dir, f'{split}.txt'), 'r') as f:
        # Read the file and split into lines, then split each line into components to form triples
        lines = f.read().split('\n')[:-1]
        triples = [row.split('\t') for row in lines]  # Each 'row' is split into head, relation, tail


# # 3. Sample triples (subject, predicate, object)
# ==================Simple Triples=====================
# triples = [
#     ("Paris", "is the capital of", "France"),
#     ("The Eiffel Tower", "is located in", "Paris"),
#     ("Berlin", "is the capital of", "Germany")
# ]

# sample_triples = triples[:50]
# =====================================================

# =================Random Triples======================
# # The list has at least 50 triples randomly to avoid ValueError
# random.seed(42)

# if len(triples) >= 50:
#     sample_triples = random.sample(triples, 50)
# else:
#     print(f"Only {len(triples)} triples available, cannot select 50.")
#     sample_triples = triples  # Optionally, use all available triples if less than 50
# =====================================================


# 4. Convert triples into sentences
triple_sentences = [' '.join(triple) for triple in triples]

# 5. Convert triples to embeddings
triple_embeddings = model.encode(triple_sentences)

# 6. Sample question
# question = "Who wrote Columbus Circle?"

questions = [
    "Who directed 'Inception'?",
    "Which movies star 'Tom Hanks'?",
    "What genre does the movie 'Mad Max: Fury Road' belong to?",
    "Name a comedy film released in 2005.",
    "Who played the lead role in 'The Matrix'?",
    "Can you list all the movies directed by 'Steven Spielberg'?",
    "What year was 'Pulp Fiction' released?",
    "Which actress won the best actress award for a movie released in 2010?",
    "Name a movie that features both 'Leonardo DiCaprio' and 'Kate Winslet'.",
    "What is the runtime of 'The Lord of the Rings: The Return of the King?",
    "Who composed the soundtrack for 'Interstellar'?",
    "List all the science fiction movies released in the 1990s.",
    "Which movies are part of the 'Harry Potter' series?",
    "Who directed the movie with the highest box office gross in 2018?",
    "What are the names of movies that deal with time travel?",
    "Can you name a film where 'Morgan Freeman' plays the president?",
    "Which movie won the Best Picture Oscar in 2003?",
    "Name a horror film directed by 'Alfred Hitchcock'.",
    "What is the sequel to 'Finding Nemo'?",
    "Who voiced 'Woody' in 'Toy Story'?"
]


results = []

for question in questions:
    # Convert question to embeddings
    question_embedding = model.encode([question])

    # Calculate cosine similarities
    similarities = cosine_similarity(question_embedding, triple_embeddings)

    # Find the index of the highest similarity score
    most_similar_index = np.argmax(similarities)
    most_similar_score = similarities[0, most_similar_index]

    # Append the results
    results.append({
        "Question": question,
        "Most Similar Triple": ' '.join(triples[most_similar_index]),
        "Answer": triples[most_similar_index][2],
        "Cosine Similarity Score": most_similar_score
    })

# Convert results to a pandas DataFrame
df_results = pd.DataFrame(results)

#%%
# Set the threshold for high similarity
threshold = 0.8

# Add a new column 'High Similarity' to the DataFrame
# It will contain 1 if the cosine similarity score is above 0.8, and 0 otherwise
df_results['High Similarity'] = (df_results['Cosine Similarity Score'] > threshold).astype(int)




# Display or save the DataFrame
df_results
# To save the DataFrame as a CSV file
# df_results.to_csv('qa_results.csv', index=False)


# # %%
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




# %%
