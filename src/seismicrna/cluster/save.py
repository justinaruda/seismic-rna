import pandas as pd

from .compare import RunOrderResults
from .io import ClusterBatchIO
from ..mask.data import MaskMutsDataset


def write_batches(dataset: MaskMutsDataset,
                  orders: list[RunOrderResults],
                  brotli_level: int):
    """ Write the cluster memberships to batch files. """
    checksums = list()
    for batch_num in dataset.batch_nums:
        resps = pd.concat((runs.best.get_resps(batch_num) for runs in orders),
                          axis=1)
        batch = ClusterBatchIO(sample=dataset.sample,
                               ref=dataset.ref,
                               sect=dataset.sect,
                               batch=batch_num,
                               resps=resps)
        _, checksum = batch.save(top=dataset.top,
                                 brotli_level=brotli_level,
                                 force=True)
        checksums.append(checksum)
    return checksums
