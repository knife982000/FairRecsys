from RecBole.recbole.data.dataset.dataset import Dataset


class InteractionSampler:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def sample(self, undersample_ratio: float = 0, oversample_ratio: float = 0):
        if undersample_ratio > 0 and oversample_ratio > 0:
            self.mixed_sample(oversample_ratio, undersample_ratio)
        elif undersample_ratio > 0:
            self.undersample(undersample_ratio)
        elif oversample_ratio > 0:
            self.oversample(oversample_ratio)
        else:
            raise ValueError("At least one of undersample_ratio or oversample_ratio must be greater than 0.")
        return self.dataset

    def mixed_sample(self, target_ratio: float, retain_ratio: float):
        if target_ratio < 1 or retain_ratio >= 1:
            raise ValueError("Target ratio must be greater than or equal to 1 and retain ratio must be less than 1.")
        return self.oversample(target_ratio).undersample(retain_ratio)

    def oversample(self, target_ratio: float):
        if target_ratio < 1:
            raise ValueError("Target ratio must be greater than or equal to 1.")
        return self.__sample(target_ratio, replace=True, inverse=False)

    def undersample(self, retain_ratio: float):
        if retain_ratio >= 1:
            raise ValueError("Retain ratio must be less than or equal to 1.")
        return self.__sample(retain_ratio, replace=False, inverse=True)

    def __sample(self, ratio: float, replace: bool, inverse: bool):
        item_frequency = self.dataset.inter_feat[self.dataset.iid_field].value_counts().to_dict()
        if inverse:
            weights = self.dataset.inter_feat[self.dataset.iid_field].map(lambda x: 1 / item_frequency[x])
        else:
            weights = self.dataset.inter_feat[self.dataset.iid_field].map(lambda x: item_frequency[x])
        n_samples = int(len(self.dataset.inter_feat) * ratio)

        sampled_data = self.dataset.inter_feat.sample(n=n_samples, weights=weights, replace=replace)
        sampled_data = sampled_data.reset_index(drop=True)

        self.dataset.inter_feat = sampled_data

        return self
