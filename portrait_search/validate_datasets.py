from pathlib import Path

from dependency_injector.wiring import Provide, inject

from portrait_search.dependencies import Container
from portrait_search.quality.dataset import load_dataset, validate_dataset

DATASETS = ["v1"]


@inject
def validate_datasets(local_data_folder: Path = Provide[Container.config.provided.local_data_folder]) -> None:
    for dataset_name in DATASETS:
        dataset = load_dataset(dataset_name)
        validate_dataset(dataset, local_data_folder / "portraits")


if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])

    validate_datasets()
