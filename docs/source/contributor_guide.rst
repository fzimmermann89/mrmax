=================
Contributor Guide
=================

Repo structure
==============
This repository uses a *pyproject.toml* file to specify all the requirements.

**.github/workflows**
    Definitions of GitHub action workflows to carry out formatting checks, run tests and automatically create this
    documentation.

**.vscode**
    Configuration files for VS Code.

**.docs**
    Files to create this documentation.

**examples**
    Python scripts showcasing how mrmax can be used. Any data needed has to be available from
    an online repository (e.g. zenodo) such that it can be automatically downloaded.
    Individual cells should be indicated with ``# %%``. For markdown cells use ``# %% [markdown]``.
    The translation from Python script to Jupyter notebook is done in pre-commit (locally and on GitHub)
    using `jupytext <https://jupytext.readthedocs.io/en/latest/>`_ . See its documentation for more details.

    After translating the scripts to notebooks, the notebooks are run and their output is converted to html and added
    to this documentation in the *Examples* section.

    All output cells in the notebooks are automatically cleared, and only cleared notebooks should be added to the repository.

**tests**
    Tests which are automatically run by pytest.
    The subfolder structure should follow the same structure as in *mrmax/src*.

**src/mrmax/algorithms**
    Everything which does something with the data, e.g. prewhiten k-space or remove oversampling.

**src/mrmax/data**
    All the data classes such as ``KData``, ``ImageData`` or ``CsmData``.
    As the names suggest these should mainly contain data and meta information.
    Any functionality beyond what is absolutely required for the classes should be put as separate functions.

**src/mrmax/operators**
    Linear and non-linear algorithms describing e.g. the transformation from image to k-space (``FourierOp``), the
    effect of receiver coils (``SensitivityOp``) or MR signal models.

**src/mrmax/phantoms**
    Numerical phantoms useful to evaluate reconstruction algorithms.

**src/mrmax/utils**
    Utilities such as spatial filters and also more basic functionality such as applying functions serially along the
    batch dimension (``smap``).


Linting
=======
We use Ruff for linting. If you are using VSCode, you can install an
`extension <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_,
which we have also added to the list of extensions that VSCode should recommend when you open the code.
We also run `mypy <https://pypi.org/project/mypy/>`_ as a type checker.

In CI, our linting is driven by `pre-commit <https://pre-commit.com/>`_.
If you install mrmax via ``pip install -e .[dev]``, pre-commit will be installed in your Python environment.
You can either add pre-commit to your git pre-commit hooks, requiring it to pass before each commit (``pre-commit install``),
or run it manually using ``pre-commit run --all-files`` after making your changes, before requesting a PR review.

Naming convention
=================
We try to follow the `PEP 8 <https://peps.python.org/pep-0008/>`_ naming convention (e.g., all lowercase variable names,
CapWords class names). We deviate for the names of source code file names containing a single class.
These are named as the class.

We try to use descriptive variable names when applicable (e.g., ``result`` instead of ``res``, ``tolerance_squared`` instead
of ``sqtol``, ``batchsize`` instead of ``m``).

A name starting with ``n_`` is used for variables describing a number of something (e.g., ``n_coils`` instead of ``ncoils`` or
``num_coils``), variable names ending with ``_op`` for operators (e.g., ``fourier_op``). We use ``img`` as a variable name
for images.

Testing
=======
We use pytest for testing. All required packages will be installed if you install mrmax via ``pip install -e .[dev]`` or ``pip install -e .[tests]``.
You can use VSCode's test panel to discover and run tests. All tests must pass before a PR can be merged. By default, we skip running CUDA tests.  You can use ``pytest -m cuda`` to run the CUDA tests if your development machine has a GPU available.

Building the Documentation
==========================
You can build the documentation locally via running ``make html`` in the ``docs`` folder. The documentation will also be build in each PR and can be viewed online.
Please check how your new additions render in the documentation before requesting a PR review.


Adding new Examples
===================
New exciting applications of mrmax can be added in ``examples`` as only ``.py`` files with code-cells. These can, for example, be used in VSCode with the Python extension, or in JupyterLab with the `jupytext <https://jupytext.readthedocs.io/en/latest/>`_ extension.
A pre-commit action will convert the scripts to notebooks. Our documetantion build will pick up these notebooks, run them, and include them with outputs in the documentation.
The data to run the examples should be publicly available and hosted externally, for example at zenodo.
Please be careful not to add any binary files to your commits.

Release Strategy
================
We are still in pre-release mode and do not guarantee a stable API / strict semantic versioning compatibility. We currently use ``0.YYMMDD`` as versioning and release in regular intervals to `pypi  <https://pypi.org/project/mrmax/>`_.

Compatibility
=============
We aim to always be compatible with the latest stable PyTorch release and the latest Python version supported by pytorch. We are compatible with one previous Python version.
Our type hints will usually only be valid with the latest PyTorch version.
