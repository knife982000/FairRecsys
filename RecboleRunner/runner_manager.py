from RecboleRunner.recbole_runner import RecboleRunner


class RunnerManager:
    _runner: RecboleRunner = None

    @classmethod
    def set_runner(cls, runner: RecboleRunner):
        cls._runner = runner

    @classmethod
    def get_runner(cls) -> RecboleRunner:
        return cls._runner
