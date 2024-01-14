import asyncio
from pathlib import Path

from dependency_injector.wiring import Provide, inject
from tabulate import tabulate
from tqdm import tqdm

from portrait_search.dependencies import Container
from portrait_search.quality.experiments import (
    all_possible_combinations_cosine_by_experiment,
    load_or_create_experiment_results,
    store_experiment_results,
)


@inject
async def do_experiments(
    container: Container,
    local_data_folder: Path = Provide[Container.config.provided.local_data_folder],
    experiment: str = Provide[Container.config.provided.experiment],
) -> None:
    if not experiment:
        raise ValueError("No experiment specified")

    results_path = local_data_folder / "experiment_results/results.json"

    results = load_or_create_experiment_results(results_path)
    experiments = all_possible_combinations_cosine_by_experiment(experiment=experiment)
    for description, judge_factory in tqdm(experiments.items(), desc="Running experiments"):
        if description in results:
            print(f"Skipping finished experiment: {description}")
            continue
        judge = judge_factory(container)
        print(f"Running experiment: {description}")
        result = await judge.evaluate()
        print(result)
        results[description] = result

    store_experiment_results(results_path, results)

    print("All experiments")
    print("---------------")
    results_table = [{"description": description, **result} for description, result in results.items()]
    results_table.sort(key=lambda x: x["precision@k"], reverse=True)  # type: ignore
    print(tabulate(results_table, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])

    asyncio.run(do_experiments(container))
