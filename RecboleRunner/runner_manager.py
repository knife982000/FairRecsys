from RecboleRunner.recbole_runner import RecboleRunner


class RunnerManager:
    _runner: RecboleRunner = None

    @classmethod
    def set_runner(cls, runner: RecboleRunner):
        cls._runner = runner
        print(f"RecboleRunner set with model: {runner.model_name}, dataset: {runner.dataset_name}")

    @classmethod
    def get_runner(cls) -> RecboleRunner:
        print("RecboleRunner retrieved: ", cls._runner)
        return cls._runner
