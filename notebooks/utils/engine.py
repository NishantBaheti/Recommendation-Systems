import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class BasicMovieRecommedation:
    """
    Basic Movie Recommendation Engine
    ----------------------------------

    Item-Item Recommendation engine. Rating based user-movie matrix and nearest neighbour clustering method.
    """
    def __init__(self, n_neighbors: int, metric: str = 'cosine'):
        self.n_neighbors: int = n_neighbors + 1
        self.metric: str = metric
        self.algorithm: str = 'brute'

    def create_two_way_maps(self, ids: Union[list, np.ndarray]) -> Tuple[dict, dict]:
        """
        Create a two way maps.

        Args:
            ids (Union[list, np.ndarray]): Ids.

        Returns:
            Tuple[dict,dict]: Map and inverse map.
        """
        id_maps = {i[0]: i[1] for i in enumerate(ids)}
        inv_id_maps = {id_maps[i]: i for i in id_maps}
        return id_maps, inv_id_maps

    def fit(self, rating_df: pd.DataFrame, movie_df: pd.DataFrame = None, user_df: pd.DataFrame = None) -> None:
        """
        Train the recommender system.

        Recommender System
        -------------------
            item item based recommendation system.

        Args:
            rating_df (pd.DataFrame): Rating data frame.
            movie_df (pd.DataFrame, optional): Movie data frame. Defaults to None.
            user_df (pd.DataFrame, optional): user data frame. Defaults to None.
        """

        all_user_ids = user_df.userId.unique() if user_df is not None \
            else rating_df.userId.unique()
        all_movie_ids = movie_df.movieId.unique() if movie_df is not None \
            else rating_df.movieId.unique()

        self.user_id_maps, self.user_inv_id_maps = self.create_two_way_maps(
            all_user_ids)
        self.movie_id_maps, self.movie_inv_id_maps = self.create_two_way_maps(
            all_movie_ids)

        user_maps = rating_df\
            .userId.apply(lambda x: self.user_inv_id_maps[x])
        movie_maps = rating_df\
            .movieId.apply(lambda x: self.movie_inv_id_maps[x])
        ratings = rating_df.rating.values

        self.rating_matrix = csr_matrix((ratings, (movie_maps, user_maps)))\
            .toarray()
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.algorithm, metric=self.metric)
        self.model.fit(self.rating_matrix)

    def get_recommendations(self, movie_id: int, n_recommendations: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Recommendations.

        Args:
            movie_id (int): Movie ID.
            n_recommendations (int): Number of recommendation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Recommended movie ids, and distances,
        """
        movie_vector = self.rating_matrix[self.movie_inv_id_maps[movie_id]].reshape(
            1, -1)
        distances, collected_neighbours = self.model.kneighbors(
            X=movie_vector, n_neighbors=n_recommendations+1, return_distance=True)

        recomm_movie_ids = np.array([i for i in map(
            lambda x: self.movie_id_maps[x], collected_neighbours[0])])
        return recomm_movie_ids, distances[0]
