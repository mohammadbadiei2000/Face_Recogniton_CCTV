import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)
        self.database = {}

    def get_embedding(self, image):
        faces = self.app.get(image)
        if len(faces) > 0:
            return faces[0].embedding
        return None

    def add_identity(self, name, embedding):
        self.database[name] = embedding

    def recognize(self, embedding, threshold=0.5):
        best_match = None
        best_score = -1

        for name, db_embedding in self.database.items():
            score = cosine_similarity(
                embedding.reshape(1, -1),
                db_embedding.reshape(1, -1)
            )[0][0]

            if score > best_score:
                best_score = score
                best_match = name

        if best_score > threshold:
            return best_match, best_score
        else:
            return "Unknown", best_score
