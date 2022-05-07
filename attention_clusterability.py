import numpy as np
from torch.utils.data import DataLoader, Dataset

from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from os import path

from sklearn.datasets import make_blobs
from torch.utils.data.dataset import T_co


def prepare_blob_dataset(hparams) -> (np.ndarray, np.ndarray):
    city_num = hparams['city_num']
    feature_dim = hparams['feature_dim']
    sample_num = hparams['sample_num']

    samples = np.zeros((sample_num, city_num, feature_dim))
    labels = np.zeros((sample_num, city_num))

    for sample in range(sample_num):
        samples[sample, :, :], labels[sample, :] = make_blobs(city_num, feature_dim)

    return samples, labels


class TorchDatasetWrapper(Dataset):
    def __init__(self, hparams):
        super(TorchDatasetWrapper, self).__init__()
        self.hparams = hparams

        self.samples, self.labels = self._generate_dataset()

    def __getitem__(self, index) -> T_co:
        sample = self.samples[index]
        label = self.labels[index]

        data_pair = {'sample': sample, 'label': label}

        return data_pair

    def __len__(self):
        return len(self.samples)

    def _generate_dataset(self):
        samples, labels = prepare_blob_dataset(self.hparams)
        return samples, labels


def main(hparams):
    # Prepare the tensorboard writer
    cur_time = datetime.now() + timedelta(hours=0)
    writer = SummaryWriter(logdir=path.join(hparams['log_dir'], cur_time.strftime("[%m-%d]%H.%M.%S")))

    # Prepare and load the training data
    dataset = TorchDatasetWrapper(hparams)
    train_iterator = DataLoader(dataset, batch_size=hparams['batch_size'], num_workers=1)

    # Instantiate the policy
    c_attention_model = AttentionModel(problem, hyper_params['feature_dim'], hyper_params['embedding_dim'],
                                       hyper_params['hidden_dim'], hyper_params['city_num'])


if __name__ == '__main__':
    # some arguments and hyperparameters
    hyper_params = {
        'num_clusters': 3,
        'feature_dim': 2,
        'city_num': 50,
        'sample_num': 1000000,
        'log_dir': 'logs_attention_cluster-ability'
    }

    main(hyper_params)
