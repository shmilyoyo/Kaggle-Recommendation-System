from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import utility


inputDataRootPath = "/data/haoxu/Data/Kaggle-Recommendation-Dataset"
outputDataRootPath = "/data/haoxu/Data/Kaggle-Recommendation-Dataset"
