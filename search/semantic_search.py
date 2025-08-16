import pickle
import openai
import numpy as np
import os
import asyncio
from collections import OrderedDict

# Get rid later
# from search.config import openai_key
# openai.api_key = openai_key

class SemanticSearch:
    def __init__(self):
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = openai_api_key
        self.model = "text-embedding-3-small"
        self.data_dir = "data"
        # Always run moderation for every query
        self.enable_moderation = True
        # small LRU cache for query embeddings to avoid duplicate OpenAI calls
        self.embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_cache_size = 1024
        self.load_data()
        

    def load_data(self):
        # loads data from pickle files into server memory
        with open(os.path.join(self.data_dir, 'embedding_matrix_32.pkl'), 'rb') as embedding_file:
            self.embedding_matrix = pickle.load(embedding_file)

        with open(os.path.join(self.data_dir, 'index_to_data_dict.pkl'), 'rb') as data_dict_file:
            self.course_data_dict = pickle.load(data_dict_file)

        with open(os.path.join(self.data_dir, 'data_to_index_dict.pkl'), 'rb') as data_to_index_file:
            self.data_to_index_dict = pickle.load(data_to_index_file)

        with open(os.path.join(self.data_dir, 'latest_sem_indices.pkl'), 'rb') as latest_semester_file:
            self.latest_semester_indices = pickle.load(latest_semester_file)
        
        with open(os.path.join(self.data_dir, 'topic_class_map.pkl'), 'rb') as topic_class_map_file:
            self.topic_class_map = pickle.load(topic_class_map_file)
        
        
        self.acad_level_to_indices_map = {}

        for level in ['Undergraduate', 'Graduate', 'Law', 'Graduate Business', 'Medical School', 'Non-Credit']:
            filename = os.path.join(self.data_dir, f"{level}_indices.pkl")
            with open(filename, 'rb') as f:
                self.acad_level_to_indices_map[level] = pickle.load(f)

        # Ensure index collections are sets for fast intersection
        if not isinstance(self.latest_semester_indices, set):
            self.latest_semester_indices = set(self.latest_semester_indices)
        for level, indices in list(self.acad_level_to_indices_map.items()):
            if not isinstance(indices, set):
                self.acad_level_to_indices_map[level] = set(indices)

        # Pre-normalize embedding matrix rows to unit vectors for faster cosine similarity
        row_norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        # avoid division by zero
        row_norms[row_norms == 0] = 1.0
        self.embedding_matrix = self.embedding_matrix / row_norms
        

    def _get_embedding_sync(self, text: str) -> np.ndarray:
        # Blocking call to OpenAI; use only via asyncio.to_thread
        clean_text = text.replace("\n", " ")
        vector = np.array(openai.Embedding.create(input=[clean_text], model=self.model)['data'][0]['embedding'], dtype=np.float32)
        # normalize to unit vector once here
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    async def _get_embedding(self, text: str) -> np.ndarray:
        cached = self.embedding_cache.get(text)
        if cached is not None:
            # refresh LRU order
            self.embedding_cache.move_to_end(text)
            return cached
        vector = await asyncio.to_thread(self._get_embedding_sync, text)
        # maintain cache size
        self.embedding_cache[text] = vector
        if len(self.embedding_cache) > self.max_cache_size:
            self.embedding_cache.popitem(last=False)
        return vector
    

    def _get_moderation_sync(self, text: str) -> bool:
        moderation_response = openai.Moderation.create(input=text)
        return moderation_response["results"][0]['flagged']

    async def get_moderation(self, text: str) -> bool:
        # Offload blocking call to a worker thread
        return await asyncio.to_thread(self._get_moderation_sync, text)


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
        # With pre-normalized rows and query, cosine similarity reduces to dot product
        similarities = np.dot(embedding_matrix, query_vector)
        return similarities


    def get_top_n_data_without_filters(self, query_vector, n=10, return_graph_data=False):
        # if there are no filters, just use the original embedding matrix
        similarities = self.cosine_similarity_search(query_vector, self.embedding_matrix)
        if n <= 0:
            return []
        # argpartition for efficiency, then sort the top-k slice
        top_n_partition = np.argpartition(similarities, -n)[-n:]
        top_n_indices = top_n_partition[np.argsort(similarities[top_n_partition])[::-1]]
        top_n_data = [self.course_data_dict[index] for index in top_n_indices]

        # add the similarity scores as values in the dictionaries
        for i in range(min(n, len(top_n_data))):
            matrix_index = top_n_indices[i]
            top_n_data[i]["similarity_score"] = similarities[matrix_index].item()
        return top_n_data


    def get_top_n_data_with_filters(self, query_vector, academic_level_filter="all", semester_filter="all", n=10, return_graph_data=False):
        filtered_embedding_matrix, original_indices = self.generate_filtered_embedding_matrix(academic_level_filter, semester_filter)
        similarities = self.cosine_similarity_search(query_vector, filtered_embedding_matrix)
        del filtered_embedding_matrix   # clear memory
        if n <= 0:
            return []
        top_n_partition = np.argpartition(similarities, -n)[-n:]
        top_n_filtered_indices = top_n_partition[np.argsort(similarities[top_n_partition])[::-1]]
        top_n_original_indices = original_indices[top_n_filtered_indices]
        top_n_data = [self.course_data_dict[index] for index in top_n_original_indices]

        # add the similarity scores as values in the dictionaries
        for i in range(min(n, len(top_n_data))):
            matrix_index = top_n_filtered_indices[i]
            top_n_data[i]["similarity_score"] = similarities[matrix_index].item()
        del similarities   # clear memory
        return top_n_data


    def get_top_n_data(self, query_vector, academic_level_filter="all", semester_filter="all", n=10, return_graph_data=False):
        if academic_level_filter == "all" and semester_filter == "all":
            return self.get_top_n_data_without_filters(query_vector, n=n, return_graph_data=return_graph_data)
        else:
            return self.get_top_n_data_with_filters(query_vector, academic_level_filter, semester_filter, n, return_graph_data=return_graph_data)


    # PCA and graph-related functionality removed; no longer used


    async def get_filtered_search_results(self, query, academic_level_filter="all", semester_filter="all", n=10, return_graph_data=False):
        # Run embedding (and optionally moderation) concurrently
        tasks = [self._get_embedding(query)]
        if self.enable_moderation:
            tasks.append(self.get_moderation(query))
        results = await asyncio.gather(*tasks)
        query_vector = results[0]
        if self.enable_moderation and len(results) > 1 and results[1]:
            return {"resultData": [], "PCATransformedQuery": None}

        top_n_data = self.get_top_n_data(query_vector, academic_level_filter=academic_level_filter, semester_filter=semester_filter, n=n, return_graph_data=False)
        response = {
            "resultData": top_n_data,
            "PCATransformedQuery": None
        }
        return response


    async def get_search_results(self, query, academic_level_filter ="all", semester_filter="all",  n=10, return_graph_data=False):
        # Backwards-compatible wrapper
        return await self.get_filtered_search_results(query, academic_level_filter, semester_filter, n, return_graph_data)
    


    def check_if_valid_course(self, mnemonic, catalog_number):
        id_tuple = (mnemonic.upper(), str(catalog_number))
        return id_tuple in self.data_to_index_dict.keys() or id_tuple in self.topic_class_map.keys()


    # method that gets called for a "similar courses" request
    def get_similar_course_results(self, mnemonic, catalog_number, academic_level_filter="all", semester_filter="all", n=10, return_graph_data=False):
        id_tuple = (mnemonic.upper(), str(catalog_number))

        # if it's a special topics course
        if id_tuple in self.topic_class_map.keys():
            results = [self.course_data_dict[self.data_to_index_dict[course]] for course in self.topic_class_map[id_tuple]]
            
            # set similarity scores to one
            for result in results:
                result["similarity_score"] = 1
            
            results.sort(key=lambda x: x["catalog_number"])

            response = {
                "resultData": results,
                "PCATransformedQuery": None
            }
            return response
        
        index = self.data_to_index_dict[id_tuple]
        # embedding_matrix rows are already normalized
        query_vector = self.embedding_matrix[index]
        top_n_data = self.get_top_n_data(query_vector, academic_level_filter=academic_level_filter, semester_filter=semester_filter, n=n, return_graph_data=return_graph_data)
        response = {
            "resultData": top_n_data,
            "PCATransformedQuery": None
        }
        return response

    async def warmup(self):
        # Prime the embedding endpoint and caches to reduce first-request latency
        try:
            await self._get_embedding("warmup query")
        except Exception:
            # Ignore warmup errors to avoid failing startup
            pass