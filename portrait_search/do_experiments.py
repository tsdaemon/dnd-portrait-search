import asyncio
from pathlib import Path

from dependency_injector.wiring import Provide, inject
from tabulate import tabulate

from portrait_search.dependencies import Container
from portrait_search.quality.experiments import EXPERIMENTS, load_or_create_experiment_results, store_experiment_results


@inject
async def do_experiments(
    container: Container, local_data_folder: Path = Provide[Container.config.provided.local_data_folder]
) -> None:
    results_path = local_data_folder / "experiment_results/results.json"

    results = load_or_create_experiment_results(results_path)
    for description, judge_factory in EXPERIMENTS.items():
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
    print(tabulate(results_table, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])

    asyncio.run(do_experiments(container))
