from BaseModel import BaseModel
import preprocessData


class ContentBasedModel(BaseModel):

    MODEL_NAME = "Content-based Model"

    def __init__(self, df):
        self.df = df