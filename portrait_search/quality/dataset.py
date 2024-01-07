from pathlib import Path

import yaml
from pydantic import BaseModel


class DatasetEntry(BaseModel):
    name: str
    path: str
    tags: list[str]
    queries: list[str]


def load_dataset(experiment: str) -> list[DatasetEntry]:
    path_to_dataset = Path(__file__).parent.parent.parent / "dataset" / experiment
    if not path_to_dataset.exists():
        raise ValueError(f"Dataset for {experiment} does not exist at path {path_to_dataset}")

    dataset_entries = []
    for yaml_file in path_to_dataset.glob("*.yaml"):
        with open(yaml_file) as file:
            yaml_data = yaml.safe_load(file)
            dataset_entry = DatasetEntry(name=yaml_file.stem, **yaml_data)
            dataset_entries.append(dataset_entry)

    return dataset_entries
