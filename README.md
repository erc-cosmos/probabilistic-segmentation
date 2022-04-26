# Arcs Bayesian Segmentation
This contains code to run a Bayesian segmentation of a sequence.

## Code organisation

The code is organised into 3 directories:
* The `bayes_arcs` folder contains generic code which is reusable and meant to be imported for final use. This includes the main algorithms and a number of presets, both for reading different data formats and for setting up the priors.
* The `notebooks` folder holds independent notebooks (in celled code format) which cover a few different use cases for the generic code. They are meant to be edited to fit the exact need.
* The `tests` folder gathers all the testing code. Testing is mainly done using the property testing paradigm (more on this below).

## Using the code
Notebooks are useful to follow along. The `single_run.py` notebook in particular shows a straightforward use case.

The main functions to call are `compute_boundary_posteriors` or `compute_both_posteriors` in the `dynamic_computation` submodule, which respectively output the posterior credence of segment end positions or a pair containing the credence for boundaries and for segments. They mainly expect 3 arguments :
* The input sequence (a numpy array, possibly with more than one column), which can be read from the databases using the functions in the `readers` submodule. Alternatively, the `synthetic_data` submodule can randomly generate data that fits some priors. If the `linear_sampling` option is `False`, it instead expects an iterable of (x,y) pairs, where x is the time associated with datapoint y (which can be multidimensional).
* The arc prior, a dictionnary whose structure matches the examples in the `default_priors` submodule.
* The length prior, which implements either the `DiscreteLengthPrior` or `ContinuousLengthPrior` classes from the `length_priors` submodule. The submodule includes a few subclasses which can be used.

The output can then be visualised using the functions in the `segment_viz` submodule (see notebooks for usage), or exported with the `writers` submodule.

The other possible entry point is using the functions in the `batch_run` sumodule. It contains a generic `batch_run` function to run the computations on a set of sequences (with the same priors), and a few presets for known collections.


## Misc notes

### Notebooks format
Notebooks are not saved in their usual `.ipynb` format, but as celled python scripts. This is so that they interface well with `git` and other code processing tools (linter, refactoring, diff, etc.), as that format does not store the output.

Visual Studio Code has good support for this format:
* They can be ran in an interactive notebook-like window using VSCode's interactive window commands (by default in the contextual menu when using the python extension)
* It can convert back to a proper `.ipynb` notebook (again in the contextual menu)

### Property-based testing
Tests use the [hypothesis](https://hypothesis.readthedocs.io/en/latest/) library under the `pytest` framework.

Relying on known output for known input is prone to blindspots and for somewhat complex computations requires circular reasoning, using the code itself to write the test it has to pass.
Instead, property-based testing checks that some generic properties are verified for the output, for arbitrarily generated input. A typical example is that output credences are indeed valid probabilities.