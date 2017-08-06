# Set up session:
import numpy as np

# Sklearn transformer interface:
from sklearn.base import BaseEstimator, TransformerMixin

class ReshapeTransformer(BaseEstimator, TransformerMixin):
    """ Reshaper
    """

    def __init__(
        self,
        shape
    ):
        self.shape = shape

    def transform(self, X):
        """ Reshape input vector:
        """
        return X.reshape(self.shape)

    def fit(self, X, y=None):
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
