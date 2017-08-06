# Set up session:
import numpy as np
import cv2

# Sklearn transformer interface:
from sklearn.base import BaseEstimator, TransformerMixin

class ColorHistogramTransformer(BaseEstimator, TransformerMixin):
    """ HOG feature extractor for grayscale input image
    """

    def __init__(
        self,
        color_space="RGB",
        bins_per_channel=8,
        intensity_range=(0, 256),
        normalize=True,
        epsilon=1e-9
    ):
        self.color_space = color_space
        self.bins_per_channel = bins_per_channel
        self.intensity_range = intensity_range
        self.normalize = normalize
        self.epsilon = epsilon

    def transform(self, X):
        """ Extract histogram for each channel
            then concatenate all channel histograms as final feature vector
        """
        return np.array([self._extract_color_histogram(x) for x in X])

    def fit(self, X, y=None):
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def _get_enhanced(self, image, color_space):
        """ Get contrast enhanced image:
        """
        if color_space == "BGR":
            image_enhanced = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image_enhanced[:, :, 2] = cv2.equalizeHist(image_enhanced[:, :, 2])
            image_enhanced = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif color_space == "RGB":
            image_enhanced = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image_enhanced[:, :, 2] = cv2.equalizeHist(image_enhanced[:, :, 2])
            image_enhanced = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            if color_space == "HSV":
                image_enhanced = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                image_enhanced[:, :, 2] = cv2.equalizeHist(image_enhanced[:, :, 2])
            elif color_space == "LUV":
                image_enhanced = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                image_enhanced[:, :, 0] = cv2.equalizeHist(image_enhanced[:, :, 0])
            elif color_space == "HLS":
                image_enhanced = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                image_enhanced[:, :, 1] = cv2.equalizeHist(image_enhanced[:, :, 1])
            elif color_space == "YUV":
                image_enhanced = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                image_enhanced[:, :, 0] = cv2.equalizeHist(image_enhanced[:, :, 0])
            elif color_space == "YCrCb":
                image_enhanced = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                image_enhanced[:, :, 0] = cv2.equalizeHist(image_enhanced[:, :, 0])

        return image_enhanced

    def _extract_color_histogram(self, image):
        """ Extract color histogram
        """
        # Convert color space
        image_enhanced = self._get_enhanced(
            image,
            self.color_space
        )

        # Get color histogram:
        features = np.concatenate(
            tuple(
                [np.histogram(image_component, self.bins_per_channel, self.intensity_range)[0] for image_component in cv2.split(image_enhanced)]
            )
        )

        # L1 normalize:
        if self.normalize:
            features = features / (features.sum() + self.epsilon)

        return features
