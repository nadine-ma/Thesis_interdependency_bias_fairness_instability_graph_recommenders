import numpy as np
from torch.utils.data import Dataset

class RatingsDataset(Dataset):
    def __init__(self, rating_data):
        super(RatingsDataset, self).__init__()
        self.ratings = rating_data

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        features = self.ratings
        data = {
            "user": features[idx][0],
            "rating": np.array(features[idx][2]).astype(np.float32),
            "item": features[idx][1],
        }
        return data
