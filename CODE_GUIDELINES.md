Code Guidelines for the QMix Package
====================================

- All new code must follow our [Python code guidelines](#python-code).
- In addition, all new contributions must include:
    - [docstrings](#docstrings) for each new module, function, class and dictionary,
    - [unit tests](#unit-tests) for each new function and class, and
    - updated [documentation](#qmix-documentation) for the QMix web page.
- Please also update the docstrings and tests if you change how a function or class operates.

Python Code
-----------

- In general, follow [Google's style guidelines](https://google.github.io/styleguide/pyguide.html) and ["Best practices for scientific computing"](http://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745) by G. Wilson et al. 
    - This includes giving variables, functions and classes meaningful names; using consistent formatting (i.e., follow what's already there); using Git effectively; using issue tracking; and writing in small steps with frequent testing.
- All Python code should also follow the [PEP-008](https://www.python.org/dev/peps/pep-0008/) guidelines.
    - Always use 4 spaces to indent code, **never tabs.**
    - Try to stay below 80 characters per line. Sometimes this doesn't make sense, so use your own judgment. 
    - The command `pep8` can be used to check PEP-008 consistency:
        - To install: `pip install pep8`
        - To run: `pep8 <filename>`
    - The command `autopep8` can be used to force PEP-008 consistency:
        - To install: `pip install autopep8`
        - To run: `autopep8 qmix --recursive --in-place --pep8-passes 2000 --verbose`
- In addition to these guidelines, please ensure that your code:
    - is compatible with Python version 3.5 and later, and
    - only depends on the standard library; the NumPy, SciPy, Numba or Matplotlib packages; and the dependencies thereof.
- Finally, please also ensure that there are only ASCII characters in your code (no unicode). 
    - Using regular expression, you can search for non-ASCII characters using `[^\x00-\x7F]+`

Docstrings
----------

- Docstrings are required for every public module, function, class and dictionary.
- Please use the [Google docstring format](https://google.github.io/styleguide/pyguide.html#381-docstrings) and follow the [PEP-257](https://www.python.org/dev/peps/pep-0257/) guidelines.
- Note: It's okay to comment private functions with inline comments (i.e., functions that start with a single underscore and are only used within that module).

Unit Tests
----------

- Unit tests are required for all public functions and classes. 
- The tests should be able to be run using `pytest` and should follow the structure seen in the `QMix/tests/` directory.
- Ensure that your new tests pass when you run:
   - ``pytest --verbose --color=yes tests/``
- You should also check your test coverage using:
   - ``pytest --verbose --color=yes --cov=qmix/ --cov-report=html tests/``

QMix Documentation
------------------

- The [QMix webpage](https://garrettj403.github.io/QMix/qmix.html) includes a section that outlines all of the modules, functions and classes that are contained within the QMix package. 
- If you add any new code, please update this documentation by running: 
   - ``sphinx-apidoc -o docs/source/ qmix/``

Other Notes
-----------

- For matrix indices, follow this order:
    - frequency (`f`), harmonic (`p`), Bessel function index (`k`), voltage point (`i`)
    - e.g., for the Thevenin voltage: `vt[f, p]`
    - e.g., for the Convolution coefficient: `ckh[f, p, k, i]`
- Use American spelling everywhere to be consistent with the rest of Python.
    - For example:
        - color, not colour (CAN/UK)
        - tunneling, not tunnelling (CAN/UK)
        - meter, not metre (CAN/UK)
    - Note: If you have Aspell installed, you can check your spelling against the American dictionary using: 
        - `aspell -c -d en_US <filename>`