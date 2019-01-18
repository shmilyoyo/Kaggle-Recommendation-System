import preprocessData

inputDataRootPath = "/data/haoxu/Data/Kaggle-Recommendation-Dataset"
outputDataRootPath = "/data/haoxu/Data/Kaggle-Recommendation-Dataset"

interactions_full_indexed_df, interactions_train_indexed_df, \
    interactions_test_indexed_df = preprocessData.mungingData(inputDataRootPath,
                                                              outputDataRootPath)

person_id = -8845298781299428018

print(preprocessData.get_items_interacted(person_id, interactions_full_indexed_df))
