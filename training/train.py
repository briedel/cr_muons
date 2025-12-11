import argparse

import numpy as np

from pathlib import Path
from dataloader import SingleHDF5Dataset, MultiHDF5Dataset, ragged_collate_fn
from normalizer import DataNormalizer
from torch.utils.data import DataLoader



def main(args):
    if len(args.infiles) == 1:
        dataset = SingleHDF5Dataset(args.infiles[0])
    if len(args.infiles) > 2:
        dataset = MultiHDF5Dataset(args.infiles)
    
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            # shuffle=True,
                            collate_fn=ragged_collate_fn)

    normalizer = DataNormalizer()

    for real_muons, batch_idx, prims, counts in dataloader:
        real_muons_norm = normalizer.normalize_features(real_muons)
        prims_norm = normalizer.normalize_primaries(prims)

        print(real_muons_norm)
        print(real_muons)
        print(prims_norm)
        print(prims)
        # train_step(real_muons_norm, batch_idx, prims_norm, counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", 
                        "--infiles", 
                        nargs='+', 
                        type=Path, 
                        required=True)
    args=parser.parse_args()

    main(args)

