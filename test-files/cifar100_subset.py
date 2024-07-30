from typing import List
from torchvision.datasets import CIFAR100


class CIFAR100Subset(CIFAR100):
    def __init__(self, subset: List[int], **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        assert max(subset) <= max(self.targets)
        assert min(subset) >= min(self.targets)

        self.aligned_indices = []
        for idx, label in enumerate(self.targets):
            if label in subset:
                self.aligned_indices.append(idx)

    def get_class_names(self):
        return [self.classes[i] for i in self.subset]

    def __len__(self):
        return len(self.aligned_indices)

    def __getitem__(self, item):
        return super().__getitem__(self.aligned_indices[item])
    
    def __getitem__(self, idx):
        img, label = super().__getitem__(self.aligned_indices[idx])
        # Remap the label to 0-29 range
        label = self.subset.index(label)
        return img, label


# if __name__ == '__main__':
#     import torchvision
#     import torchvision.transforms as transforms
#     from torch.utils.data import DataLoader
#     import matplotlib.pyplot as plt

#     minimal_transform = transforms.Compose([transforms.ToTensor()])
#     cifar100_subset = CIFAR100Subset(
#         subset=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
#         root='./dataset',
#         train=True,
#         download=True,
#         transform=minimal_transform
#     )

#     print(cifar100_subset.get_class_names())
#     print(len(cifar100_subset))

#     dataloader = DataLoader(cifar100_subset, batch_size=64, shuffle=True)
#     x, _ = next(iter(dataloader))

#     grid_img = torchvision.utils.make_grid(x, nrow=8)
#     plt.imshow(grid_img.permute(1, 2, 0))
#     plt.show()