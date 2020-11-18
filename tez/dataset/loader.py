import torch


class TezDataLoader:
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(self)
        self.dataset = dataset

    def fetch(self, **kwargs):
        data_loader = torch.utils.data.DataLoader(self.dataset, **kwargs)
        return data_loader
