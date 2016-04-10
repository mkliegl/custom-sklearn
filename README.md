# Custom scikit-learn estimators and transformers

Repository for some custom [scikit-learn][sklearn] estimators
and transformers.

The documentation is viewable [here][doc].


## Contents

* [flexible_linear.py](flexible_linear.py) implements a [scikit-learn][sklearn]
  estimator for linear regression with custom training and
  regularization cost functions.
* [heavytail.ipynb](heavytail.ipynb) is a [Jupyter][jupyter] notebook demonstrating
  the use of [flexible_linear.py](flexible_linear.py) to improve
  predictions in the presence of heavy-tailed noise.



## Installation

This is not a formal package. If anything looks useful to you,
please feel free to clone or fork the repository, or just copy
individual files.

The documentation is written using [Sphinx][sphinx].
To build the documentation locally:
```bash
pip install sphinx sphinx_rtd_theme
( cd docs ; make html )
```
To view the documentation, browse to `docs/_build/html/index.html`.
Alternatively, checkout the `gh-pages` branch or view the documentation
online [here][doc].


## License

MIT. See the [LICENSE](LICENSE) file for more details.


[doc]:     https://mkliegl.github.io/custom-sklearn/
[sklearn]: http://scikit-learn.org/
[jupyter]: http://jupyter.org
[sphinx]:  http://www.sphinx-doc.org/
