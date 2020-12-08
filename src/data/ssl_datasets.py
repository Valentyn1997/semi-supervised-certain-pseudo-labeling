import torchvision
import numpy as np
import math
from src import DATA_PATH
from os.path import exists
from os import mkdir
import logging
from torchvision.datasets import VisionDataset
from copy import deepcopy
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def getitem_wrapper(getitem):
    def getitem_with_ind(self, index):
        return getitem(self, index) + (index, )
    return getitem_with_ind


class SLLDatasetsCollection:
    """Contains the collection of train, val and test datasets"""
    def __init__(self, source: str, n_labelled: int, val_ratio: float, random_state: int,
                 train_l_transform: object, train_ul_transform: object, test_transform: object):
        self.dataset_path = f'{DATA_PATH}/{source}'
        self.n_labelled = n_labelled
        mkdir(self.dataset_path) if not exists(self.dataset_path) else None
        self.Dataset = getattr(torchvision.datasets, source)
        if not hasattr(self.Dataset, 'new_getitem'):  # Disable multiple wrapping while multi-run
            self.Dataset.__getitem__ = getitem_wrapper(self.Dataset.__getitem__)
            self.Dataset.new_getitem = True

        if source in ['CIFAR10', 'CIFAR100']:
            self.train_dataset = self.Dataset(root=self.dataset_path, train=True, download=True)
            self.test_dataset = self.Dataset(root=self.dataset_path, train=False, download=True)
        elif source in ['SVHN']:
            self.train_dataset = self.Dataset(root=self.dataset_path, split='train', download=True)
            self.test_dataset = self.Dataset(root=self.dataset_path, split='test', download=True)
        elif source in ['STL10']:
            self.train_dataset = self.Dataset(root=self.dataset_path, split='train+unlabeled', download=True)
            self.test_dataset = self.Dataset(root=self.dataset_path, split='test', download=True)

        self.train_dataset, self.val_dataset = self.split_train_val(self.train_dataset, val_ratio, random_state)
        self.train_l_dataset, self.train_ul_dataset = self.remove_labels_and_split(self.train_dataset, n_labelled, shuffle=True)
        self.classes = self.test_dataset.classes

        # Setting transforms
        self.train_dataset.transform = test_transform
        self.train_l_dataset.transform = train_l_transform
        self.train_ul_dataset.transform = train_ul_transform
        self.test_dataset.transform = test_transform
        if val_ratio > 0.0:
            self.val_dataset.transform = test_transform

    @staticmethod
    def split_train_val(dataset: VisionDataset, val_ratio: float, random_state: int):
        labels_attr = None
        if hasattr(dataset, 'labels'):
            labels_attr = 'labels'
        elif hasattr(dataset, 'targets'):
            labels_attr = 'targets'

        labels = np.array(getattr(dataset, labels_attr))

        train_set = deepcopy(dataset)

        if val_ratio > 0.0:
            val_set = deepcopy(dataset)
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
            for train_index, val_index in splitter.split(dataset.data, labels):
                train_set.data = dataset.data[train_index]
                setattr(train_set, labels_attr, labels[train_index])
                val_set.data = dataset.data[val_index]
                setattr(val_set, labels_attr, labels[val_index])

            return train_set, val_set
        else:
            return train_set, None

    @staticmethod
    def remove_labels_and_split(dataset: VisionDataset, n_labels_to_leave: int, shuffle=False):
        """
        Removing labels for SSL purposes
        @param shuffle: Shuffle indices to remove
        @param dataset: Dataset to change
        @param n_labels_to_leave: int
        """
        labels_attr = None
        if hasattr(dataset, 'labels'):
            labels_attr = 'labels'
        elif hasattr(dataset, 'targets'):
            labels_attr = 'targets'

        labels = np.array(getattr(dataset, labels_attr))
        old_labels = np.copy(labels)
        assert n_labels_to_leave <= len(labels)  # Requiring more labels, than exist

        classes = np.unique(labels)
        classes = classes[classes != -1]  # For originally SSL datasets
        n_labels_per_cls = n_labels_to_leave // len(classes)

        if n_labels_to_leave % len(classes) != 0:
            logger.warning(f'Not equal distribution of classes. Used number of labels: {n_labels_per_cls * len(classes)}')

        for c in classes:
            if shuffle:
                n_to_drop = len(labels[labels == c]) - n_labels_per_cls
                ind_to_drop = np.random.choice(np.arange(0, len(labels[labels == c])), n_to_drop, replace=False)
                labels[np.where(labels == c)[0][ind_to_drop]] = -1
            else:
                labels[np.where(labels == c)[0][n_labels_per_cls:]] = -1

        labeled_set = deepcopy(dataset)
        labeled_set.data = labeled_set.data[labels != -1]
        setattr(labeled_set, labels_attr, list(labels[labels != -1]))
        setattr(labeled_set, 'old_labels', list(old_labels[labels != -1]))  # For compatibility

        unlabeled_set = deepcopy(dataset)
        unlabeled_set.data = unlabeled_set.data[labels == -1]
        setattr(unlabeled_set, labels_attr, list(old_labels[labels == -1]))
        setattr(unlabeled_set, 'old_labels', list(old_labels[labels == -1]))

        return labeled_set, unlabeled_set


class FixMatchCompositeTrainDataset(VisionDataset):
    """Wrapper, to generate composite item (labelled, unlabelled)"""
    def __init__(self, l_dataset: VisionDataset, ul_dataset: VisionDataset, mu: int, size='max',
                 val_dataset: VisionDataset = None):
        self.l_dataset = l_dataset
        self.ul_dataset = ul_dataset
        self.val_dataset = val_dataset
        self.mu = mu

        self.l_indexes = []
        self.ul_indexes = []
        self.val_indexes = []
        if isinstance(size, int):
            self.len = size
        else:
            self.len = eval(size)(len(self.l_dataset), len(self.ul_dataset) // self.mu)
        self.construct_indices()

    def __getitem__(self, i):
        if self.val_dataset is not None:
            return [self.l_dataset[ind] for ind in self.l_indexes[i]], \
                   [self.ul_dataset[ind] for ind in self.ul_indexes[i]], \
                   [self.val_dataset[ind] for ind in self.val_indexes[i]]
        else:
            return [self.l_dataset[ind] for ind in self.l_indexes[i]], [self.ul_dataset[ind] for ind in self.ul_indexes[i]]

    def construct_indices(self):
        l_indices = np.arange(0, len(self.l_dataset))
        ul_indices = np.arange(0, len(self.ul_dataset))
        if self.val_dataset is not None:
            val_indices = np.arange(0, len(self.val_dataset))

        l_n_repeats = math.ceil(self.len / len(l_indices))
        ul_n_repeats = math.ceil(self.len * self.mu / len(ul_indices))
        if self.val_dataset is not None:
            val_n_repeats = math.ceil(self.len / len(val_indices))

        l_indices_repeated = np.tile(l_indices, l_n_repeats).reshape(l_n_repeats, l_indices.size)
        ul_indices_repeated = np.tile(ul_indices, ul_n_repeats).reshape(ul_n_repeats, ul_indices.size)
        if self.val_dataset is not None:
            val_indices_repeated = np.tile(val_indices, val_n_repeats).reshape(val_n_repeats, val_indices.size)

        self.l_indexes = np.array(list(map(np.random.permutation, l_indices_repeated))).flatten()[:self.len]
        self.ul_indexes = np.array(list(map(np.random.permutation, ul_indices_repeated))).flatten()[:self.len * self.mu]
        if self.val_dataset is not None:
            self.val_indexes = np.array(list(map(np.random.permutation, val_indices_repeated))).flatten()[:self.len]

        self.l_indexes = self.l_indexes.reshape(self.len, 1)
        self.ul_indexes = self.ul_indexes.reshape(self.len, self.mu)
        if self.val_dataset is not None:
            self.val_indexes = self.val_indexes.reshape(self.len, 1)

    def __len__(self):
        # will be called every epoch
        self.construct_indices()
        return self.len
