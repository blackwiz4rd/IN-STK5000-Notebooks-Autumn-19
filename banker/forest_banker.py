from banker import BankerBase, run
from random import choice

# my imports
import numpy as np
# model for fitting dataset
from sklearn.ensemble import RandomForestClassifier
# select best model
from sklearn.model_selection import cross_val_score

class ForestBanker(BankerBase):
    """ForestBanker implementation."""

    def __init__(self, interest_rate):
        self.interest_rate = interest_rate

    """
    This function uses a random forest classifier to predict new probabilities
    """
    def fit(self, X, y):
        # find out which n_depth number is the best for the random
        # forest classifier (as we were doing for K-NN algorithm)
        self.best_max_depth = 10
        self.clf = RandomForestClassifier(n_estimators=100,  random_state=0, max_depth=self.best_max_depth) # storing classifier
        self.clf.fit(X, y)

    def predict_proba(self, x):
        ## order is class 1 = good_loan_proba, 2 = bad_loan_proba
        # print("classes", self.clf.classes_)
        # prediction = self.clf.predict_proba(np.reshape(x.to_numpy(), (1, -1)))
        prediction = self.clf.predict_proba(x)
        # print("prediction", prediction)
        return prediction[0][1]

    def expected_utility(self, X, action):
        amount_of_loan = X['amount']
        length_of_loan = X['duration']
        if action == 1:
            # print("amount gained lost", amount_of_loan*(1 + self.rate*length_of_loan), -amount_of_loan)
            # print("expected_utility grant probability ", amount_of_loan*(1 + self.rate*length_of_loan) * (1-self.predict_proba(x)) - amount_of_loan * self.predict_proba(x), (1-self.predict_proba(x)))
            # return amount_of_loan*(1 + self.rate*length_of_loan) * (1-self.predict_proba(x)) - amount_of_loan * self.predict_proba(x)
            gained = amount_of_loan*(pow(1 + self.interest_rate, length_of_loan) - 1) * (1 - self.predict_proba(x))
            lost = amount_of_loan * self.predict_proba(x)
            return gained - lost

        return 0

    def get_best_action(self, X):
        actions = [0, 1]
        best_action = -np.inf
        best_utility = -np.inf
        for a in actions:
            utility_a = self.expected_utility(X, a)
            if utility_a > best_utility:
                best_action = a
                best_utility = utility_a
        print("grant = ", best_action)
        return best_action


if __name__ == '__main__':
    run()
