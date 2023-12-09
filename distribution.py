import numpy as np


class LaplaceDistribution:
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        median = np.median(x, axis=0)
        return np.mean(np.abs(x - median), axis=0)

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        #эта часть верна
        self.loc = np.median(features, axis=0)
        self.scale = np.mean(np.abs(features - self.loc), axis = 0)

    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        log_prob = np.log(1 / (2 * self.scale)) - np.abs(values - self.loc) / self.scale
        return log_prob

    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        # return np.exp(self.logpdf(value))
        prob = np.exp(self.logpdf(values))
        return prob
