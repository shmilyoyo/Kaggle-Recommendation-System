from BaseModel import BaseModel
import preprocessData
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentBasedModel(BaseModel):

    MODEL_NAME = "Content-based Model"

    def __init__(self, model_id, contents_ids, users_profiles, maxtrix, items_df=None):
        """Initilaize the data used in this classself.
        
        Arguments:
            model_id {int} - the model id
            contents_ids {DataFrame} -- contents ids in original articles
            users_profiles {dict} -- mapping person_id to profile vector
            maxtrix {matrix} -- the matrix returned from vectorizer
            items_df {DataFrame} -- the default additional items added in result (default: {None})
        """

        self.contents_ids = contents_ids
        self.users_profiles = users_profiles
        self.maxtrix = maxtrix
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def runModel(self):
        pass

    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        """Get similar items to the user profile based on cosine similarityself.
        
        Arguments:
            person_id {int} -- the person id
        
        Keyword Arguments:
            topn {int} -- the number of most similar items (default: {1000})
        
        Returns:
            list -- a ranked list of most similar items to the user profile
        """

        cosine_similarities = cosine_similarity(
            self.users_profiles[person_id], self.maxtrix)
        similar_indexes = cosine_similarities.argsort().flatten()[-topn:]
        similar_items = sorted([(self.contents_ids[i], cosine_similarities[0, i])
                                for i in similar_indexes], key=lambda x: -x[1])

        return similar_items

    def recommend_items(self, person_id, items_to_ignore=[], topn=10,
                        verbose=False):
        """Recommend items to person with person_idself.
        
        Arguments:
            person_id {int} -- person id
        
        Keyword Arguments:
            items_to_ignore {list} -- a list of items that user has already visited (default: {[]})
            topn {int} -- the number of final recommendation (default: {10})
            verbose {bool} -- indicate whether add items extra information (default: {False})
        
        Raises:
            Exception -- if want to use verbose mode, the items_df should be provided
        
        Returns:
            DataFrame -- a recommendation dataframe
        """

        similar_items = self._get_similar_items_to_user_profile(person_id)
        # filter out the items that are already interacted
        similar_items_filter = list(
            filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filter, columns=[
                                          "contentId", "recStrength"]).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception("items_df is required in verbose mode.")

            recommendations_df = recommendations_df.merge(self.items_df,
                                                          how="left",
                                                          left_on="contentId",
                                                          right_on="contentId"
                                                          )[["recStrength",
                                                             "contentId",
                                                             "title",
                                                             "url",
                                                             "lang"]]
        return recommendations_df
