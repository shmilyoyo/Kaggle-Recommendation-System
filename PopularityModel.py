from BaseModel import BaseModel


class PopularityModel(BaseModel):

    MODEL_NAME = "Popularity"

    def __init__(self, model_id, popularity_df, items_df):
        self.model_id = model_id
        self.popularity_df = popularity_df
        self.items_df = items_df

    def runModel(self):
        pass

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        recommendation_df = self.popularity_df[~self.popularity_df["contentId"].isin(
            items_to_ignore)].sort_values("eventStrength", ascending=False).head(topn)
        
        if verbose:
            if self.items_df is None:
                raise Exception("item_df is required in verbose mode")
            
            recommendation_df = recommendation_df.merge(self.items_df, how="left", left_on="contentId", right_on="contentId")[["eventStrength", "contentId", "title", "url", "lang"]]

        return recommendation_df