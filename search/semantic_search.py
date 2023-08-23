import pickle
import openai
import numpy as np
import json
import os


# Get rid later
# from search.config import openai_key
# openai.api_key = openai_key




class SemanticSearch:
    def __init__(self):
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = openai_api_key
        self.model = "text-embedding-ada-002"
        with open('search/embedding_matrix.pkl', 'rb') as embedding_file:
            self.embedding_matrix = pickle.load(embedding_file)
        
        with open('search/course_data_dict.pkl', 'rb') as data_dict_file:
            self.course_data_dict = pickle.load(data_dict_file)


    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], model=self.model)['data'][0]['embedding']
    

    def cosine_similarity_search(self, query_vector, embedding_matrix):
        similarities = np.dot(self.embedding_matrix, query_vector) / (np.linalg.norm(self.embedding_matrix, axis=1) * np.linalg.norm(query_vector))
        return similarities


    def get_top_results_json(self, query, n=10):
        query_vector = self.get_embedding(query)
        similarities = self.cosine_similarity_search(query_vector, self.embedding_matrix)

        top_n_indices = np.argsort(similarities)[::-1][:n]
        top_n_data = [self.course_data_dict[index] for index in top_n_indices]

        # add the similarity scores as values in the dictionaries
        for i in range(n):
            matrix_index = top_n_indices[i]
            top_n_data[i]["similarity_score"] = similarities[matrix_index]

        return json.dumps(top_n_data)

