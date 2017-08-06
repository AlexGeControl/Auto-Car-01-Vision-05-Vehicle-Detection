# Set up session:
import numpy as np
import cv2

# Sklearn transformer interface:
from sklearn.base import BaseEstimator, TransformerMixin

class TemplateTransformer(BaseEstimator, TransformerMixin):
    """ Template extractor
    """
    def __init__(
        self,
        color_space="HSV",
        template_size=(16, 16),
        normalize=False,
        epsilon=1e-9
    ):
        self.color_space = color_space
        self.template_size = template_size
        self.normalize = normalize
        self.epsilon = epsilon

    def transform(self, X):
        """ Extract histogram for each channel
            then concatenate all channel histograms as final feature vector
        """
        return np.array([self._extract_template(x) for x in X])

    def fit(self, X, y=None):
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def _get_channel_shape(self, image, color_space):
        """ Get image component for shape representation:
        """
        if color_space == "BGR":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif color_space == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            if color_space == "HSV":
                return cv2.split(
                    cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                )[2]
            elif color_space == "LUV":
                return cv2.split(
                    cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                )[0]
            elif color_space == "HLS":
                return cv2.split(
                    cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                )[1]
            elif color_space == "YUV":
                return cv2.split(
                    cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                )[0]
            elif color_space == "YCrCb":
                return cv2.split(
                    cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                )[0]
            else:
                return None

    def _extract_template(self, image):
        """ Extract template
        """
        # Convert color space and extract component for shape representation:
        image_channel_shape = self._get_channel_shape(image, self.color_space)
        image_channel_shape = cv2.equalizeHist(image_channel_shape)

        # Get color histogram:
        features = cv2.resize(
            image_channel_shape,
            self.template_size,
            interpolation = cv2.INTER_AREA
        ).ravel()

        # L2 normalize:
        if self.normalize:
            features = features / np.sqrt(np.sum(features ** 2) + self.epsilon)

        return features
