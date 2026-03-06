from datasets import Dataset as HFDataset


def concat_dataset_to_hf(dataset) -> HFDataset:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    first = dataset[0]
    return HFDataset.from_dict({key: [row[key] for row in dataset] for key in first.keys()})
