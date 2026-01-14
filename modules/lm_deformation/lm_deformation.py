import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkDeformationModel(nn.Module):
    def __init__(self, landmark_dim=204, num_emotions=8, emotion_embedding_dim=16, hidden_dim=256):
        super(LandmarkDeformationModel, self).__init__()

        # Emotion embedding
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)

        # Input dimension: neutral landmarks + emotion embedding
        input_dim = landmark_dim + emotion_embedding_dim

        # Regression network
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, landmark_dim)
        )
    
    def adjust_dimensions(self, a, b):
        if a.dim() < b.dim():
            # Add dimensions to 'a' until it matches 'b'
            while a.dim() < b.dim():
                a = a.unsqueeze(0)
        elif a.dim() > b.dim():
            # Add dimensions to 'b' until it matches 'a'
            while a.dim() > b.dim():
                b = b.unsqueeze(0)
        return a, b



    def forward(self, neutral_landmarks, emotion_labels, delta=1):
        """
        neutral_landmarks: Tensor of shape [B, landmark_dim]
        emotion_labels: Tensor of shape [B] (LongTensor)
        """
        # Get emotion embeddings
        emotion_embed = self.emotion_embedding(emotion_labels) 
        emotion_embed = emotion_embed.squeeze(1)
        emotion_embed = emotion_embed.repeat(neutral_landmarks.shape[0], 1)


        x = torch.cat([neutral_landmarks, emotion_embed], dim=1)  # [B, landmark_dim + emotion_embedding_dim]

        # Predict landmark deformation
        delta_landmarks = self.regressor(x)  # [B, landmark_dim]

        # Compute emotional landmarks
        emotional_landmarks = neutral_landmarks + delta_landmarks * delta  

        return emotional_landmarks
