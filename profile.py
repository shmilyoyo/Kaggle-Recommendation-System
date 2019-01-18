import scipy
import numpy as np
import sklearn


class Profile:

    def __init__(self, content_ids, matrix, interactions_df, articles):
        """Initialize the content ids and data matrix return from vectorizer.

        Arguments:
            content_ids {list} -- a list of content ids from original articles
            matrix {matrix} -- the matrix returned by vectorizer
            interactions_df {DataFrame} -- the DataFrame of the interactions
            articles {DataFrame} -- the original article DataFrame
        """

        self.content_ids = content_ids
        self.matrix = matrix
        self.interactions_df = interactions_df
        self.articles_df = articles

    def get_item_profile(self, item_id):
        """Get the vector corresponding to item_id in matrix.

        Arguments:
            item_id {int} -- the item id in matrix

        Returns:
            np.array -- the vector in the matrix
        """

        idx = self.content_ids.index(item_id)
        item_profile = self.matrix[idx:idx+1]

        return item_profile

    def get_items_profiles(self, item_ids):
        """Get a list of vectors corresponding to item_ids in matrix.

        Arguments:
            item_ids {list} -- a list of ids need to be got from matrix

        Returns:
            sparse matrix -- the total vectors corresponding to item_ids
        """

        items_profiles_list = [self.get_item_profile(
            item_id) for item_id in item_ids]
        items_profiles = scipy.sparse.vstack(items_profiles_list)

        return items_profiles

    def build_user_profile(self, person_id, interactions_indexed_df):
        """Build a user profile based on the normalization.

        normalize the items profiles for person with person_id based on the content strength

        Arguments:
            person_id {int} -- the person index
            interactions_indexed_df {DataFrame} -- interactions DataFrame indexed on personId

        Returns:
            matrix -- a user normalized profile
        """

        interactions_person_df = interactions_indexed_df.loc[person_id]
        user_items_profiles = self.get_items_profiles(
            interactions_person_df["contentId"])

        # average weight the matrix
        user_items_strengths = np.array(
            interactions_person_df["eventStrength"]).reshape(-1, 1)
        user_items_strengths_weighted_avg = np.sum(user_items_profiles.multiply(
            user_items_strengths), axis=0) / np.sum(user_items_strengths)
        user_profile_norm = sklearn.preprocessing.normalize(
            user_items_strengths_weighted_avg)

        return user_profile_norm

    def build_users_profiles(self):
        """Build all users profilesself.

        Returns:
            dict -- mapping personId to corresponding items profiles
        """

        # intersect the contentIds in interactions dataframe and articles dataframe
        interactions_indexed_df = self.interactions_df[
            self.interactions_df["contentId"].isin(
                self.articles_df["contentId"])].set_index("personId")

        user_profiles = {}
        for person_id in interactions_indexed_df.index.unique():
            user_profiles[person_id] = self.build_user_profile(
                person_id, interactions_indexed_df)

        return user_profiles
