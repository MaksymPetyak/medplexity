from medplexity.benchmarks.loaders import Loader


class MockLoader(Loader):
    """Simple mock loader that will return on load the given to it object"""

    def __init__(self, mock_output):
        self.output = mock_output

    def load(self, *args, **kwargs):
        return self.output
