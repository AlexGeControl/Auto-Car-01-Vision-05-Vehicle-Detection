# Set up session:
import numpy as np
import cv2

# Skimage hog descriptor:
from skimage.feature import hog

# Sklearn transformer interface:
from sklearn.base import BaseEstimator, TransformerMixin

class HOGTransformer(BaseEstimator, TransformerMixin):
    """ HOG feature extractor for grayscale input image
    """

    def __init__(
        self,
        color_space="YUV",
        shape_only=False,
        orientations = 9,
        pixels_per_cell = (4, 4),
        cells_per_block = (2, 2),
        transform_sqrt = False,
        block_norm = "L1"
    ):
        self.color_space = color_space
        self.shape_only = shape_only
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.transform_sqrt = transform_sqrt
        self.block_norm = block_norm

    def transform(self, X):
        """ Extract HOG feature for given grayscale input image X
        """
        if self.shape_only:
            return np.array([self._extract_shape_hog(x) for x in X])
        else:
            return np.array([self._extract_all_channel_hog(x) for x in X])

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

    def _extract_shape_hog(self, image):
        """ Extract HOG description
        """
        # Convert color space and extract component for shape representation:
        image_channel_shape = self._get_channel_shape(image, self.color_space)
        image_channel_shape = cv2.equalizeHist(image_channel_shape)

        # Extract HOG description:
        features = hog(
            image_channel_shape,
            orientations = self.orientations,
            pixels_per_cell = self.pixels_per_cell,
            cells_per_block = self.cells_per_block,
            transform_sqrt = self.transform_sqrt,
            block_norm = self.block_norm,
            visualise = False
        )

        return features

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

    def _extract_all_channel_hog(self, image):
        """ Extract HOG description
        """
        # Convert color space and extract component for shape representation:
        image_enhanced = self._get_enhanced(
            image,
            self.color_space
        )

        # Extract HOG description:
        features = []
        for image_channel in cv2.split(image_enhanced):
            features.append(
                hog(
                    image_channel,
                    orientations = self.orientations,
                    pixels_per_cell = self.pixels_per_cell,
                    cells_per_block = self.cells_per_block,
                    transform_sqrt = self.transform_sqrt,
                    block_norm = self.block_norm,
                    visualise = False
                )
            )

        return np.concatenate(tuple(features))
