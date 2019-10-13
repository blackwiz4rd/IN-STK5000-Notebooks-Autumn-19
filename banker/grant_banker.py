from banker import BankerBase, run
from random import choice


class GrantBanker(BankerBase):
    """DeterministicBanker implementation."""

    def __init__(self, interest_rate):
        self.interest_rate = interest_rate

    def fit(self, X, y):
        pass

    def get_best_action(self, X):
        return 1


if __name__ == '__main__':
    run()
