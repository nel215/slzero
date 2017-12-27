from slzero.solve import solve


class SLZeroSparseCoder(object):

    def __init__(self, dictionary):
        self.dictionary_ = dictionary
        self.components_ = None

    def fit(self, X):
        self.components_ = solve(self.dictionary_, X)
