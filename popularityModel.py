from BaseModel import BaseModel


class PopularityModel(BaseModel):

    MODEL_NAME = "Popularity"

    def __init__(self, model_id, item_popularity_df, items_df=None):
        """Initialize the data needed in this class.
        
        Arguments:
            model_id {int} -- the model id
            item_popularity_df {DataFrame} -- the item popularity dataframe
            items_df {DataFrame} -- the default items added in the result (default: {None})
        """

        self.model_id = model_id
        self.item_popularity_df = item_popularity_df
        self.items_df = items_df

    def runModel(self):
        pass

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, person_id, items_to_ignore=[],
                       topn=10, verbose=False):
        """Recommend the items.
        
        Arguments:
            person_id {int} -- the person id
        
        Keyword Arguments:
            items_to_ignore {list} -- a list of items that people has visited (default: {[]})
            topn {int} -- the number of recommendation (default: {10})
            verbose {bool} -- indicate whether add items extra information to result (default: {False})
        
        Raises:
            Exception -- if want to use verbose mode, the items_df should be provided
        
        Returns:
            DataFrame -- a recommendation dataframe
        """

        recommendation_df = self.item_popularity_df[~self.item_popularity_df[
            "contentId"].isin(items_to_ignore)].sort_values("eventStrength",
                                                            ascending=False
                                                            ).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception("item_df is required in verbose mode")

            recommendation_df = recommendation_df.merge(self.items_df, how="left",
                                                        left_on="contentId",
                                                        right_on="contentId")[
                                                            ["eventStrength",
                                                             "contentId", "title",
                                                             "url", "lang"]]

        return recommendation_df
