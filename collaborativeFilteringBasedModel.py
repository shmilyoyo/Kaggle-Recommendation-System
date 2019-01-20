from BaseModel import BaseModel


class CFRecommender(BaseModel):

    MODEL_NAME = "Collaborative Filtering based Model"

    def __init__(self, model_id, cf_predictions_df, items_df=None):
        """Initialize the data needed in this class.

        Arguments:
            model_id {str} -- the model id
            cf_predictions_df {DataFrame} -- the item popularity dataframe
            items_df {DataFrame} -- the default items added in the result (default: {None})
        """

        self.model_id = model_id
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def runModel(self):
        pass

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(
            ascending=False).reset_index().rename(columns={user_id: "recStrength"})
        recommendations_df = sorted_user_predictions[~sorted_user_predictions["contentId"].isin(
            items_to_ignore)].sort_values("recStrength", ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception("items_df is required in verbose mode.")

            recommendations_df = recommendations_df.merge(self.items_df, how="left",
                                                          left_on="contentId",
                                                          right_on="contentId"
                                                          )[["recStrength",
                                                             "contentId", "title",
                                                             "url", "lang"]]

        return recommendations_df
