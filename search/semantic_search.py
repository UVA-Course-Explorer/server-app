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
        self.load_data()


    def load_data(self):
        # loads data from pickle files into server memory
        data_dir = "data"
        with open(os.path.join(data_dir, 'embedding_matrix_32.pkl'), 'rb') as embedding_file:
            self.embedding_matrix = pickle.load(embedding_file)

        with open(os.path.join(data_dir, 'index_to_data_dict.pkl'), 'rb') as data_dict_file:
            self.course_data_dict = pickle.load(data_dict_file)

        with open(os.path.join(data_dir, 'data_to_index_dict.pkl'), 'rb') as data_to_index_file:
            self.data_to_index_dict = pickle.load(data_to_index_file)

        self.acad_level_to_indices_map = {}

        for level in ['Undergraduate', 'Graduate', 'Law', 'Graduate Business', 'Medical School', 'Non-Credit']:
            filename = os.path.join(data_dir, f"{level}_indices.pkl")
            with open(filename, 'rb') as f:
                self.acad_level_to_indices_map[level] = pickle.load(f)
        
        with open(os.path.join(data_dir, "latest_sem_indices.pkl"), 'rb') as f:
            self.latest_semester_indices = pickle.load(f)


    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return np.array(openai.Embedding.create(input = [text], model=self.model)['data'][0]['embedding'], dtype=np.float32)
    

    def generate_filtered_embedding_matrix(self, academic_level_filter, semester_filter):
        original_indices = set([i for i in range(len(self.embedding_matrix))])
   
        if academic_level_filter != "all":
            original_indices &= self.acad_level_to_indices_map[academic_level_filter]
   
        if semester_filter == "latest":
            original_indices &= self.latest_semester_indices

        original_indices = np.array(list(original_indices))
        filtered_embedding_matrix = self.embedding_matrix[original_indices]
        return filtered_embedding_matrix, original_indices


    def cosine_similarity_search(self, query_vector, embedding_matrix):
        similarities = np.dot(embedding_matrix, query_vector) / (np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(query_vector))
        return similarities


    def get_top_n_data_without_filters(self, query_vector, n=10):
        # if there are no filters, just use the original embedding matrix
        similarities = self.cosine_similarity_search(query_vector, self.embedding_matrix)
        top_n_indices = np.argsort(similarities)[::-1][:n]
        top_n_data = [self.course_data_dict[index] for index in top_n_indices]

        # add the similarity scores as values in the dictionaries
        for i in range(n):
            matrix_index = top_n_indices[i]
            top_n_data[i]["similarity_score"] = similarities[matrix_index].item()
        return top_n_data


    def get_top_n_data_with_filters(self, query_vector, academic_level_filter="all", semester_filter="all", n=10):
        filtered_embedding_matrix, original_indices = self.generate_filtered_embedding_matrix(academic_level_filter, semester_filter)
        similarities = self.cosine_similarity_search(query_vector, filtered_embedding_matrix)
        del filtered_embedding_matrix   # clear memory
        top_n_filtered_indices = np.argsort(similarities)[::-1][:n]
        top_n_original_indices = original_indices[top_n_filtered_indices]
        top_n_data = [self.course_data_dict[index] for index in top_n_original_indices]

        # add the similarity scores as values in the dictionaries
        for i in range(min(n, len(top_n_data))):
            matrix_index = top_n_filtered_indices[i]
            top_n_data[i]["similarity_score"] = similarities[matrix_index].item()
        del similarities   # clear memory
        return top_n_data


    def get_top_n_data(self, query_vector, academic_level_filter="all", semester_filter="all", n=10):
        if academic_level_filter == "all" and semester_filter == "all":
            return self.get_top_n_data_without_filters(query_vector, n=n)
        else:
            return self.get_top_n_data_with_filters(query_vector, academic_level_filter, semester_filter, n)


    def get_search_results(self, query, academic_level_filter ="all", semester_filter="all",  n=10):
        query_vector = self.get_embedding(query, model=self.model)
        top_n_data = self.get_top_n_data(query_vector, academic_level_filter=academic_level_filter, semester_filter=semester_filter, n=n)
        return top_n_data


    def check_if_valid_course(self, mnemonic, catalog_number):
        id_tuple = (mnemonic, str(catalog_number))
        return id_tuple in self.data_to_index_dict.keys()

    
    def get_similar_course_results(self, mnemonic, catalog_number, academic_level_filter="all", semester_filter="all", n=10):
        id_tuple = (mnemonic, str(catalog_number))
        if not id_tuple in self.data_to_index_dict.keys():
            return []  # no matching courses
        index = self.data_to_index_dict[id_tuple]
        query_vector = self.embedding_matrix[index]
        top_n_data = self.get_top_n_data(query_vector, academic_level_filter=academic_level_filter, semester_filter=semester_filter, n=n+1)
        return top_n_data[1:]