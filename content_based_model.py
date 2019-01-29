from base_model import BaseModel

class ContentedBasedModel(BaseModel):

    def __init__(self, model_id):
        self.model_id = model_id
    
    def runModel(self):
        pass
    
    def get_items_profiles(self):
        pass
    
    def get_user_profile(self):
        pass
    
    def load_user_profile(self):
        pass
    
    def update_user_profile(self):
        pass

    def get_score_of_items(self):
        pass
    
    def recommend_items(self):
        pass