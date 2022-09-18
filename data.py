from imports import *
from args import *

def get_data(args):

    transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
            ]
        )

    train_dataset = None
    valid_dataset = None
    test_dataset = None

    unique, counts = np.unique(train_dataset.targets, return_counts=True)
    weights = torch.Tensor(sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=unique, y=np.asarray(train_dataset.targets)))

    return (train_dataset, valid_dataset, test_dataset), weights

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.data = None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    args = TrainingArgs()
    data, weights = get_data(args)

    args.data = data
