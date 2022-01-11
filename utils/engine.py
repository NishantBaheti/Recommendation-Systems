import numpy as np
import pandas as pd
from typing import Union
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class BasicMovieRecommedation:
    """
    Basic Movie Recommendation Engine

    rating based user-movie matrix and nearest neighbour
    """

    def __init__(self, n_recomms:int, metric:str = 'cosine'):
        self.n_neighbors:int = n_recomms + 1
        self.metric:str = metric
        self.algorithm:str = 'brute'

    def create_two_way_maps(self, ids:Union[list, np.ndarray]):
        
        id_maps = {i[0]: i[1] for i in enumerate(ids)}
        inv_id_maps = {id_maps[i]: i for i in id_maps}
        return id_maps, inv_id_maps

    def fit(self, rating_df:pd.DataFrame, movie_df:pd.DataFrame = None, user_df:pd.DataFrame=None):

        all_user_ids = user_df.userId.unique() if user_df is not None else rating_df.userId.unique()
        all_movie_ids = movie_df.movieId.unique() if movie_df is not None else rating_df.movieId.unique()

        self.user_id_maps, self.user_inv_id_maps = self.create_two_way_maps(
            all_user_ids)
        self.movie_id_maps, self.movie_inv_id_maps = self.create_two_way_maps(
            all_movie_ids)

        user_maps = rating_df.userId.apply(lambda x: self.user_inv_id_maps[x])
        movie_maps = rating_df.movieId.apply(lambda x: self.movie_inv_id_maps[x])
        ratings = rating_df.rating.values

        self.rating_matrix = csr_matrix((ratings, (movie_maps, user_maps))).toarray()
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.algorithm, metric=self.metric)
        self.model.fit(self.rating_matrix)

    def get_recommendations(self, movie_id:int):
        movie_vector = self.rating_matrix[self.movie_inv_id_maps[movie_id]].reshape(1, -1)
        distances, collected_neighbours = self.model.kneighbors(
            X=movie_vector, n_neighbors=self.n_neighbors, return_distance=True)

        recomm_movie_ids = [i for i in map(
            lambda x: self.movie_id_maps[x], collected_neighbours[0])]
        return recomm_movie_ids, distances[0]
