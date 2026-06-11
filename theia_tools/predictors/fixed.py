import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CosinePredictor:
    def __init__(self, memory):
        self.memory = memory
 
    def predict(self, image, model):
        # find NN wrt cos_sim of CLIP embeddings
        result_dict = {}
        image_features = model(image).numpy().flatten()
        representatives = [np.array(representative[model.feature_name]).flatten() for representative in self.memory] 
        cos_sim = cosine_similarity([image_features], representatives)[0]
        nearest_neighbor = self.memory[np.argmax(cos_sim)]
        result_dict = {
                'category_id': nearest_neighbor['category_id'],
                'category_name': self.memory.category_map[str(nearest_neighbor['category_id'])],
                'similarity': cos_sim[np.argmax(cos_sim)]
        }
        return result_dict
