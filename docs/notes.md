The batching process in the ParquetBatchIterableDataset is handled primarily by PyArrow and the Dataset itself, rather than the standard PyTorch DataLoader collation loop.

Here is the breakdown of who splits the data:

1. Vectorized Splitting (Standard Mode)
In normal streaming mode, the splitting is done by PyArrow using the iter_batches method.
Inside the _iter_parquet function (line parquet_dataset.py:145):

pf.iter_batches(batch_size=self.batch_size, ...) is called.
PyArrow reads a chunk of rows (a "Record Batch") matching your batch_size directly from the file.
The code then performs vectorized operations on that chunk to create the flat_muons, batch_idx, prims, and counts tensors.
2. Manual Splitting (Global Shuffle Mode)
If shuffle=True is set (line parquet_dataset.py:211):

The dataset loads all events from all files into memory first.
It performs a global permutation of the events.
It then manually slices the shuffled data in a loop (line parquet_dataset.py:253):

for i in range(0, num_events, self.batch_size):    # Slices the tensors into batches of size batch_size
3. The Role of the DataLoader
Because the dataset yields pre-batched tensors, the DataLoader in muon_datamodule.py is configured with batch_size=None.

This tells PyTorch: "Do not try to batch these items or call a collate function; the dataset is already giving me complete batches." This is significantly faster than the default PyTorch behavior because it avoids the overhead of moving individual small Python objects around.