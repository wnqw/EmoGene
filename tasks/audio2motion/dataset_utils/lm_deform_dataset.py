import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tasks.audio2motion.dataset_utils.mead_dataset import MEADSeqDataset
from utils.commons.indexed_datasets import IndexedDataset


class MEADLandmarkDeformationDataset(MEADSeqDataset):
    def __init__(self, prefix='train'):
        self.pairs = []  # Initialize pairs before superclass init
        super().__init__(prefix)
        # Initialize self.ds if it's None
        if self.ds is None:
            self.ds = IndexedDataset(f'{self.ds_path}/{self.db_key}')
        self.build_pairs()

    def build_pairs(self):
        """
        Build pairs of neutral and emotional landmarks for the same subject and utterance.
        """
        subject_utterances = {}

        # Organize data by subject and utterance
        for idx in range(len(self.ds)):
            item = self._get_item(idx)
            if item is None:
                continue
            item_id = item['item_id']
            _, subject_id, emotion_code, intensity, utterance = item_id.split('_')

            emotion = self.emo_map[emotion_code]
            key = f"{subject_id}_{utterance}"

            if key not in subject_utterances:
                subject_utterances[key] = {}
            subject_utterances[key][emotion] = idx

        # Build pairs
        for key, emotions_dict in subject_utterances.items():
            if 'neutral' in emotions_dict:
                neutral_idx = emotions_dict['neutral']
                for emotion, idx in emotions_dict.items():
                    if emotion != 'neutral':
                        self.pairs.append((neutral_idx, idx))
        # print(f"Total pairs: {len(self.pairs)}")
        # print(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        neutral_idx, emotion_idx = self.pairs[idx]
        # print(neutral_idx, emotion_idx)
        neutral_item = self._get_item(neutral_idx)
        emotion_item = self._get_item(emotion_idx)

        if neutral_item is None or emotion_item is None:
            return None

        # Process neutral item
        neutral_landmarks = self.process_item(neutral_item)
        # # Process emotional item
        emotional_landmarks = self.process_item(emotion_item)

        text_emotion = [self.emo_map[emotion_item['emotion']]]
        int_emo_encoded = self.label_encoder.transform(text_emotion)
        emotion_label = torch.from_numpy(int_emo_encoded) # int tensor


        return {
            'neutral_landmarks': neutral_landmarks,
            'emotional_landmarks': emotional_landmarks,
            'emotion_label': emotion_label
        }

    def process_item(self, item):
        """
        Process an item to extract the landmarks.
        """
        t_lm, dim_lm, _ = item['idexp_lm3d'].shape # [T, 68, 3]
        landmarks = torch.from_numpy(item['idexp_lm3d']).reshape(t_lm, -1).float()[0]
        # print(landmarks.shape)

        # landmarks = item['idexp_lm3d']
        # print('landmarks:', landmarks.shape)    

        return landmarks

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return None

        neutral_landmarks = torch.stack([s['neutral_landmarks'] for s in samples], dim=0)  # [B, landmark_dim]
        emotional_landmarks = torch.stack([s['emotional_landmarks'] for s in samples], dim=0)  # [B, landmark_dim]
        emotion_labels = torch.tensor([s['emotion_label'] for s in samples], dtype=torch.long)  # [B]

        return {
            'neutral_landmarks': neutral_landmarks,
            'emotional_landmarks': emotional_landmarks,
            'emotion_labels': emotion_labels
        }

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=0):
        loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=self.collater, num_workers=num_workers)
        return loader

if __name__ == '__main__':
    ds = MEADLandmarkDeformationDataset(prefix='train')
    # Optionally, you can test the loader
    loader = ds.get_dataloader()
    for batch in loader:
        print(batch)
        break
