import torch
from torch.utils.data import Dataset


class TurbulenceDataset(Dataset):
    def __init__(self, w_fields, psi_fields, days_ahead=1):
        # Initialize data (should be standardized)
        self.w_fields = torch.from_numpy(w_fields).to(torch.float32)
        self.psi_fields = torch.from_numpy(psi_fields).to(torch.float32)
        self.days_ahead = days_ahead

    def __len__(self):
        return self.w_fields.shape[0] - self.days_ahead

    def __getitem__(self, idx):
        input = torch.stack((self.w_fields[idx, :, :], self.psi_fields[idx, :, :]))
        target = self.w_fields[idx + self.days_ahead, :, :]
        return input, target
