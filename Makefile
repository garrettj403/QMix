# Makefile for QMix
#
# Makefile examples that I drew from:
# https://github.com/pjz/mhi/blob/master/Makefile
# http://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects.html
#

QMIX_PATH=./qmix/
TEST_PATH=./tests/

all: test clean_all

# Run tests (with Py.Test) ---------------------------------------------------

test:
	pytest --verbose --color=yes $(TEST_PATH) 
	echo "Ignore FutureWarnings... this comes from the SciPy package."

test_cov:
	pytest --verbose --color=yes --cov=$(QMIX_PATH) $(TEST_PATH)
	echo "Ignore FutureWarnings... this comes from the SciPy package."

test_report: 
	pytest --verbose --color=yes --cov=$(QMIX_PATH) --cov-report=html $(TEST_PATH)
	open htmlcov/index.html
	echo "Ignore FutureWarnings... this comes from the SciPy package."

# Clean ----------------------------------------------------------------------

clean_test:
	find . -name 'htmlcov' -exec rm -rf {} +

clean_bytecode:
	find . -name '*.pyc' -exec rm -f {} +

clean_hidden:
	find . -name '.coverage' -exec rm -rf {} +
	find . -name '.cache' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean: clean_bytecode clean_hidden clean_test

# Misc -----------------------------------------------------------------------

.PHONY: all test clean_test clean clean_bytecode clean_hidden
