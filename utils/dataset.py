import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from utils.data_loader import load_all_genes, load_train_genes, load_test_genes
from utils.histone_loader import HISTONE_MODS, get_bw_data, str_to_idx


def get_gene_unique(gene: pd.Series) -> str:
    """
    Returns a unique string representation for given gene information.

    :param gene: Series object including cell_line and gene_name.
    :return: string representing given gene
    """
    return f'{gene.cell_line}_{gene.gene_name}'


def list_2d_to_np(array: list[list]) -> np.ndarray:
    return np.array([np.array(e) for e in array])


class HistoneDataset:

    def __init__(self,
                 genes: pd.DataFrame,
                 histone_mods: list[str] = None,
                 window_size: int = 5000,
                 bin_size: int = 100,
                 bin_value_type: str = 'mean') -> None:
        """
        DataSet for model training based on histone modification data alone.
        Load histone modification signal averages or pre-generate if missing.

        :param genes: DataFrame of gene information from CAGE-train, including cell_line and gex for train genes
        :param histone_mods: list of histone modification signal types to look at
        :param window_size: number of nucleotides around the TSS start to look at (must be even!)
        :param bin_size: length of sequence to average histone modification values over
        :param bin_value_type: method how to average bin values
        """
        assert window_size % 2 == 0

        self.genes = genes
        self.histone_mods = histone_mods
        if histone_mods is None:
            self.histone_mods = HISTONE_MODS
        self.window_size = window_size
        self.n_bins = window_size // bin_size
        self.bin_value_type = bin_value_type

        self.histones = self.load_histone_data()
        self.sequences = None

    def __len__(self) -> int:
        return len(self.genes)

    def __getitem__(self, idx) -> (np.ndarray, np.ndarray):
        gene = self.genes.iloc[idx, :]

        features = self.histones[get_gene_unique(gene)]
        # idk why simply to_numpy() couldn't process inner lists..
        features = list_2d_to_np(features)  # shape (batch_size, nr_histones, nr_bins)
        if self.sequences is not None:
            # seq data shape: (batch_size, left_flank + right flank, 4)
            features = features, list_2d_to_np(self.sequences[get_gene_unique(gene)])
        if 'gex' not in gene:
            return features
        return features, gene.gex

    @staticmethod
    def get_xy(genes_df):
        dataset = HistoneDataset(genes_df)
        all_features, all_gex = [], []
        for idx in range(len(dataset.genes)):
            features, gex = dataset.__getitem__(idx)
            all_features.append(features)
            all_gex.append(gex)
        return np.array(all_features), np.array(all_gex)

    @staticmethod
    def get_x(genes_df, histones=None):
        if histones is None:
            histones = HISTONE_MODS
        dataset = HistoneDataset(genes_df, histone_mods=histones)
        all_features = []
        for idx in range(len(dataset.genes)):
            features = dataset.__getitem__(idx)
            if len(features) == 2:
                features = features[0]
            all_features.append(features)
        return np.array(all_features)

    def load_histone_data(self):
        histone_file = f'../data/histones_w{self.window_size}_b{self.n_bins}_{self.bin_value_type}.pkl'
        if not os.path.exists(histone_file):
            df = self.get_data()
            df.to_pickle(histone_file)
            print(f'Saved data to {histone_file}')
        df = pd.read_pickle(histone_file)
        return df.iloc[str_to_idx(self.histone_mods)]

    def get_data(self) -> pd.DataFrame:
        """
        Generates histone modification data by bins for each gene.
        """
        flank_size = self.window_size // 2

        print(f'Generating pkl file with histone and chromatin accessibility data...')
        all_genes = load_all_genes()
        data_per_gene = {}
        for i in tqdm(range(len(all_genes))):
            gene = all_genes.iloc[i, :]
            start = gene.TSS_start - flank_size
            end = gene.TSS_start + flank_size - 1  # marks last nucleotide index

            features = get_bw_data(gene.cell_line, gene.chr, start, end, value_type=self.bin_value_type,
                                   n_bins=self.n_bins)
            data_per_gene[get_gene_unique(gene)] = features
        return pd.DataFrame.from_dict(data_per_gene)


def get_split(test_size: float = 0.2, train_cell_line: int = None):
    df = load_train_genes()
    groups = np.array(df.chr)

    gss = GroupShuffleSplit(n_splits=1, train_size=1 - test_size, random_state=42)

    (train_idx, test_idx) = next(gss.split(X=df, y=None, groups=groups))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    if train_cell_line is not None:
        assert train_cell_line in [1, 2]
        train_df = train_df[train_df.cell_line == train_cell_line]
        test_cell_line = 2 if train_cell_line == 1 else 1
        test_df = test_df[test_df.cell_line == test_cell_line]

    x_train, y_train = HistoneDataset.get_xy(train_df)
    x_test, y_test = HistoneDataset.get_xy(test_df)

    return (x_train, y_train), (x_test, y_test)


def get_cv_splits():
    test_size = 0.2
    df = load_train_genes()
    groups = np.array(df.chr)

    split_data = []
    gss = GroupShuffleSplit(n_splits=5, train_size=1 - test_size, random_state=42)

    for (train_idx, test_idx) in gss.split(X=df, y=None, groups=groups):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        x_train, y_train = HistoneDataset.get_xy(train_df)
        x_test, y_test = HistoneDataset.get_xy(test_df)

        split_data.append(((x_train, y_train), (x_test, y_test)))

    return split_data


def get_final_split():
    df = load_train_genes()
    x_train, y_train = HistoneDataset.get_xy(df)

    test_df = load_test_genes()
    x_test = HistoneDataset.get_x(test_df)
    return (x_train, y_train), (x_test, test_df)


if __name__ == '__main__':
    get_cv_splits()
