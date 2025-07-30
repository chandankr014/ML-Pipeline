from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer, RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
np.random.seed(42)



class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Transformer that clips outliers in each column to specified quantile bounds.

    Parameters
    ----------
    lower : float, default=0.05
        Lower quantile to use for clipping.
    upper : float, default=0.95
        Upper quantile to use for clipping.

    Attributes
    ----------
    bounds : dict
        Dictionary mapping column names to (lower_bound, upper_bound) tuples.

    Methods
    -------
    fit(X, y=None)
        Computes the quantile bounds for each column.
    transform(X)
        Clips values in each column to the learned quantile bounds.
    """
    def __init__(self, lower=0.05, upper=0.95):
        self.lower = lower; self.upper = upper; self.bounds = {}
    def fit(self, X, y=None):
        for c in X.columns:
            self.bounds[c] = (X[c].quantile(self.lower), X[c].quantile(self.upper))
        return self
    def transform(self, X):
        X = X.copy()
        for c, (low, high) in self.bounds.items():
            X[c] = X[c].clip(low, high)
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies the natural logarithm (log1p) to input data.

    This transformer is suitable for use in sklearn pipelines. It applies
    np.log1p (i.e., log(1 + x)) to all values in the input array or DataFrame,
    which is useful for stabilizing variance and normalizing skewed data.

    Methods
    -------
    fit(X, y=None)
        Returns self; does not learn any parameters.
    transform(X)
        Applies np.log1p to the input data X.
    """
    def fit(self, X, y=None): return self
    def transform(self, X): return np.log1p(X)


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorizer transformer for numerical data.
    Clips values of each column to the given lower and upper quantiles.
    Suitable for use in sklearn pipelines.
    """
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        self.bounds_ = {}
        for c in X.columns:
            lo = X[c].quantile(self.lower)
            hi = X[c].quantile(self.upper)
            self.bounds_[c] = (lo, hi)
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X


class SeasonalRegressorImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in columns using a linear regression model
    fitted on a time-related column (e.g., 'month'). For each column with
    missing values (excluding the time column), fits a LinearRegression
    using the time column as the predictor and imputes missing values
    based on the predicted values from the model.

    Parameters
    ----------
    time_col : str, default='month'
        The name of the column to use as the time or seasonal predictor.

    Attributes
    ----------
    models : dict
        Dictionary mapping column names to their fitted LinearRegression models.
    """
    def __init__(self, time_col='month'):
        self.time_col = time_col
        self.models = {}

    def fit(self, X, y=None):
        X = X.copy()
        time = X[self.time_col]

        for col in X.columns:
            if col == self.time_col:
                continue

            if X[col].isna().sum() > 0:
                notnull_mask = X[col].notnull()
                model = LinearRegression()
                model.fit(time[notnull_mask].values.reshape(-1, 1), X[col][notnull_mask])
                self.models[col] = model

        return self

    def transform(self, X):
        X = X.copy()
        time = X[self.time_col]

        for col, model in self.models.items():
            nan_mask = X[col].isna()
            if nan_mask.any():
                preds = model.predict(time[nan_mask].values.reshape(-1, 1))
                X.loc[nan_mask, col] = preds

        return X


class Gaussianizer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that applies a power transformation to make features more Gaussian.

    This transformer wraps sklearn.preprocessing.PowerTransformer and supports both the Yeo-Johnson and Box-Cox methods.
    It is useful for stabilizing variance, minimizing skewness, and making data more suitable for modeling by transforming
    features to resemble a normal distribution.

    Parameters
    ----------
    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        The power transformation method to use. 'yeo-johnson' works with both positive and negative values,
        while 'box-cox' requires strictly positive values.
    **kw : dict
        Additional keyword arguments passed to sklearn.preprocessing.PowerTransformer.

    Attributes
    ----------
    pt_ : sklearn.preprocessing.PowerTransformer
        The fitted PowerTransformer instance.
    """
    def __init__(self, method='yeo-johnson', **kw):
        self.method = method
        self.kw = kw

    def fit(self, X, y=None):
        self.pt_ = PowerTransformer(method=self.method, **self.kw)
        self.pt_.fit(X)
        return self

    def transform(self, X):
        return self.pt_.transform(X)


class OrdinalBinner(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that discretizes continuous features into ordinal bins.

    This transformer wraps sklearn.preprocessing.KBinsDiscretizer and bins each feature into a specified number of ordinal levels (0 to n_bins-1).
    Useful for converting continuous variables into categorical ordinal features.

    Parameters
    ----------
    n_bins : int, default=5
        The number of bins to produce.
    encode : {'ordinal', 'onehot', 'onehot-dense'}, default='ordinal'
        Method used to encode the transformed result.
    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used to define the widths of the bins.

    Attributes
    ----------
    bin_ : sklearn.preprocessing.KBinsDiscretizer
        The fitted KBinsDiscretizer instance.
    """
    def __init__(self, n_bins=5, encode='ordinal', strategy='quantile'):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

    def fit(self, X, y=None):
        self.bin_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode=self.encode,
            strategy=self.strategy
        )
        self.bin_.fit(X)
        return self

    def transform(self, X):
        return self.bin_.transform(X)


class VarianceFilter(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that removes features with variance less than or equal to a specified threshold.

    This transformer wraps sklearn.feature_selection.VarianceThreshold and is useful for feature selection by eliminating features that do not vary enough across samples.

    Parameters
    ----------
    threshold : float, optional (default=0.01)
        Features with a training-set variance less than or equal to this threshold will be removed.

    Attributes
    ----------
    vt_ : sklearn.feature_selection.VarianceThreshold
        The fitted VarianceThreshold instance.
    """
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.vt_ = VarianceThreshold(threshold=self.threshold)
        self.vt_.fit(X)
        return self

    def transform(self, X):
        return self.vt_.transform(X)


class RobustScalerTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that centers features by their median and scales them according to the interquartile range (IQR).

    This transformer is useful for robustly scaling features that may contain outliers. It wraps sklearn.preprocessing.RobustScaler.

    Parameters
    ----------
    **kw : dict
        Additional keyword arguments passed to sklearn.preprocessing.RobustScaler.

    Attributes
    ----------
    scaler_ : sklearn.preprocessing.RobustScaler
        The fitted RobustScaler instance.
    """
    def __init__(self, use_standard_scaler: bool = False, **kw):
        self.use_standard_scaler = use_standard_scaler
        self.kw = kw

    def fit(self, X, y=None):
        if self.use_standard_scaler:
            self.scaler_ = StandardScaler(**self.kw)
        else:
            self.scaler_ = RobustScaler(**self.kw)
        self.scaler_.fit(X)
        return self

    def transform(self, X):
        return self.scaler_.transform(X)


class PCAReducer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer for Principal Component Analysis (PCA) dimensionality reduction.

    Parameters
    ----------
    n_components : int, float, None or str, optional (default=None)
        Number of components to keep. If None, all components are kept.
        See sklearn.decomposition.PCA for details.

    **kw : dict
        Additional keyword arguments passed to sklearn.decomposition.PCA.

    Attributes
    ----------
    pca_ : sklearn.decomposition.PCA
        The fitted PCA instance.
    """
    def __init__(self, n_components=None, **kw):
        self.n_components = n_components
        self.kw = kw

    def fit(self, X, y=None):
        self.pca_ = PCA(n_components=self.n_components, **self.kw)
        self.pca_.fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)
