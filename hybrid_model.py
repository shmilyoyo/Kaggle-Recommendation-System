from base_model import BaseModel


class HybridModel(BaseModel):

    MODEL_NAME = "Collaborative Filtering based Model"

    def __init__(self, model_id, cb_rec_model, cf_rec_model, items_df=None):
        """Initialize the data needed in this class.

        Arguments:
            model_id {str} -- the model id
            cf_predictions_df {DataFrame} -- the item popularity dataframe
            items_df {DataFrame} -- the default items added in the result (default: {None})
        """

        self.model_id = model_id
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def runModel(self):
        pass

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        cb_recs_df = self.cb_rec_model.recommend_items(
            user_id, items_to_ignore, 1000, verbose).rename(
                columns={"recStrength": "recStrengthCB"})

        cf_recs_df = self.cf_rec_model.recommend_items(
            user_id, items_to_ignore, 1000, verbose).rename(
                columns={"recStrength": "recStrengthCF"})

        recs_df = cb_recs_df.merge(
            cf_recs_df, how="inner", left_on="contentId", right_on="contentId")

        recs_df["recStrengthHybrid"] = recs_df["recStrengthCB"] * \
            recs_df["recStrengthCF"]

        recommendations_df = recs_df.sort_values(
            "recStrengthHybrid", ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception("items_df is required in verbose mode")

            recommendations_df = recommendations_df.merge(self.items_df,
                                                          how="left",
                                                          left_on="contentId",
                                                          right_on="contentId"
                                                          )[["recStrengthHybrid",
                                                             "contentId", "title",
                                                             "url", "lang"]]

        return recommendations_df
