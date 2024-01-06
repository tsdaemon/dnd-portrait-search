import asyncio

from tabulate import tabulate

from portrait_search.quality.experiments import EXPERIMENTS


async def do_experiments() -> None:
    results_table = []
    for description, config_method in EXPERIMENTS.items():
        judge = config_method()
        print(f"Running experiment: {description}")
        result = await judge.evaluate()
        print(result)
        results_for_table = {"description": description}
        results_for_table.update(result.items())  # type: ignore
        results_table.append(results_for_table)

    print("All experiments")
    print("---------------")
    print(tabulate(results_table, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    asyncio.run(do_experiments())
