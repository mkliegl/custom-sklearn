# Author:  Markus Kliegl
# License: MIT
"""Regularized linear regression with custom training and regularization costs.

Linear regression estimator that allows specification of arbitrary
training and regularization cost functions.

For a linear model:

.. math::

   \\textrm{predictions} = X \\cdot W

this model attempts to find :math:`W` by minimizing:

.. math::

   \\min_{W} \\left\\{
       \\textrm{cost}(X \cdot W - y) + C \\cdot \\textrm{reg_cost}(W)
   \\right\\}

for given training data :math:`X, y`. Here :math:`C` is the
regularization strength and :math:`\\textrm{cost}` and
:math:`\\textrm{reg_cost}` are customizable cost functions
(e.g., the :math:`\\ell^2` or :math:`\\ell^1` norms).

*Note:* In reality, we fit an intercept (bias coefficient) as well.
Think of :math:`X` in the above as having an extra column of 1's.

Ideally, the cost functions should be convex and continuously
differentiable.

We provide some cost functions - see the :data:`cost_func_dict`
dictionary. If you want to use a custom cost function, it should
be of the form::

    def custom_cost_func(z, **opts):
        # <code to compute cost and gradient>
        return cost, gradient

where `cost` is a float, `gradient` is an array of the same
dimensions as `z`, and you may specify any number of keyword
arguments. See :func:`l1_cost_func`, :func:`l2_cost_func`,
:func:`japanese_cost_func` for examples.
"""
import numpy as np
import scipy.optimize
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def l2_cost_func(z):
    """Squared :math:`\\ell^2` cost and gradient

    .. math::

        \\mathrm{cost}(z) = \\frac12 ||z||_{\\ell^2}^2
        = \\frac12 \sum_i |z_i|^2 \,.

    Args:
        z (ndarray): Input vector.

    Returns:
        Tuple[float, ndarray]: The cost and gradient (same shape as `z`).
    """
    return 0.5 * np.dot(z, z), z


def l1_cost_func(z):
    """:math:`\\ell^1` cost and gradient

    .. math::

        \\mathrm{cost}(z) = ||z||_{\\ell^1} = \\sum_i |z_i| \,.


    .. note::

       This cost is not differentiable. For a smooth
       alternative, see :func:`japanese_cost_func`.

    Args:
        z (ndarray): Input vector.

    Returns:
        Tuple[float, ndarray]: The cost and gradient (same shape as `z`).
    """
    return np.sum(np.abs(z)), np.sign(z)


def japanese_cost_func(z, eta=0.1):
    """'Japanese bracket' cost and gradient

    Computes cost and gradient for the cost function:

    .. math::

        \\mathrm{cost}(z) = \\eta^2 \\sum_i \\left(
            \\sqrt{ 1 + \\left( \\frac{z_i}{\\eta} \\right)^2 } - 1
        \\right) \,.

    This cost function interpolates componentwise between the
    :math:`\\ell^2` and squared :math:`\\ell^1` norms
    and is thus useful for reducing the impact of outliers
    (or when dealing with heavy-tailed rather than Gaussian noise).

    The key to understanding this is that the *Japanese bracket*

    .. math::
        \\langle z \\rangle := \\sqrt{ 1 + |z|^2 }

    satisfies these asymptotics:

    .. math::

        \\sqrt{ 1 + |z|^2 } - 1 \\approx \\begin{cases}
            \\frac12 |z|^2 & \\text{for $|z| \\ll \\eta$}
            \\\\ |z| & \\text{for $|z| \\gg \\eta$}
        \\end{cases} \,.

    Args:
        z (ndarray): Input vector.
        eta (Optional[float]): Positive scale parameter.

    Returns:
        Tuple[float, ndarray]: The cost and gradient (same shape as `z`).
    """
    z_norm = z / eta
    z_jap = np.sqrt(1.0 + z_norm * z_norm)  # componentwise Japanese bracket
    cost = eta**2 * np.sum(z_jap - 1.0)
    gradient = z / z_jap
    return cost, gradient

cost_func_dict = {
    'l2': l2_cost_func,
    'l1': l1_cost_func,
    'japanese': japanese_cost_func,
}
"""Dictionary of implemented cost functions."""


class FitError(Exception):
    """Exception raised when fitting fails.

    Attributes:
        message (str): Error message.
        res (scipy.optimize.OptimizeResult): Results returned by
            `scipy.optimize.minimize`. See SciPy documentation on
            `OptimizeResult`_ for details.

    .. _OptimizeResult: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html  # NOQA
    """

    def __init__(self, message, res):
        self.message = message
        self.res = res


class FlexibleLinearRegression(BaseEstimator):
    """Regularized linear regression with custom training/regularization costs.

    Args:
        C (Optional[float]): Nonnegative regularization
            coefficient. (Zero means no regularization.)
        cost_func (Optional[callable or str]): Training cost
            function. If not callable, should be one of 'l1', 'l2',
            or 'japanese'.
        cost_opts (Optional[dict]): Parameters to pass to
            `cost_func`.
        reg_cost_func (Optional[callable or str]): Regularization
            cost function. If not callable, should be one of 'l1',
            'l2', or 'japanese'.
        reg_cost_opts (Optional[dict]): Parameters to pass to
            `reg_cost_func`.

    Attributes:
        coef_ (ndarray): Weight matrix of shape (n_features+1,).
            (coef_[0] is the intercept coefficient.)
    """

    def __init__(
            self, C=1.0, cost_func='l2', cost_opts=None,
            reg_cost_func='l2', reg_cost_opts=None):
        self.C = C
        self.cost_func = cost_func
        self.cost_opts = cost_opts
        self.reg_cost_func = reg_cost_func
        self.reg_cost_opts = reg_cost_opts

    def _check_cost_func(self, cost_func, cost_opts):
        if not callable(cost_func):
            try:
                cost_func = cost_func_dict[cost_func]
            except KeyError:
                raise ValueError(
                    "Unknown cost function: '{}'".format(cost_func))

        if cost_opts is None:
            cost_opts = {}

        return cost_func, cost_opts

    def fit(self, X, y):
        """Fit the model.

        Args:
            X (ndarray): Training data of shape ``(n_samples, n_features)``.
            y (ndarray): Target values of shape ``(n_samples,)``.

        Returns:
            self

        Raises:
            FitError: If the fitting failed.
        """
        X, y = check_X_y(X, y, y_numeric=True)
        C = self.C
        cost_func, cost_opts = self._check_cost_func(
            self.cost_func, self.cost_opts)
        reg_cost_func, reg_cost_opts = self._check_cost_func(
            self.reg_cost_func, self.reg_cost_opts)

        # add a column of ones to X (for intercept coefficient)
        X = np.hstack((np.ones((X.shape[0], 1), dtype=float), X))

        def objective(W):
            # compute training cost/grad
            cost, outer_grad = cost_func(np.dot(X, W) - y, **cost_opts)
            grad = np.dot(outer_grad, X)  # chain rule

            # add regularization cost/grad (but don't regularize intercept)
            reg_cost, reg_grad = reg_cost_func(W[1:], **reg_cost_opts)
            cost += C * reg_cost
            grad[1:] += C * reg_grad

            return cost, grad

        initial_coef_ = np.zeros(X.shape[1])
        res = scipy.optimize.minimize(
            objective, initial_coef_, jac=True, method='L-BFGS-B')
        if res.success:
            self.coef_ = res.x
        else:
            raise FitError("Fit failed: {}".format(res.message), res=res)

        return self

    def predict(self, X):
        """Predict using the model.

        Args:
            X (ndarray): Data of shape ``(n_samples, n_features)``.

        Returns:
            y (ndarray): Predicted values of shape ``(n_samples,)``.
        """
        check_is_fitted(self, 'coef_')
        X = check_array(X)
        n_features = self.coef_.shape[0] - 1
        if X.shape[1] != n_features:
            raise ValueError(
                "X should have %d features, not %d" % (n_features, X.shape[1]))

        y = np.dot(X, self.coef_[1:]) + self.coef_[0]

        return y


def test_estimator():
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(FlexibleLinearRegression)


def test_gradients():
    z0 = np.random.randn(5)
    for cost_func in cost_func_dict.values():
        func = lambda z: cost_func(z)[0]
        grad = lambda z: cost_func(z)[1]
        assert scipy.optimize.check_grad(func, grad, z0) < 1e-5


if __name__ == '__main__':
    import nose2
    nose2.main()
