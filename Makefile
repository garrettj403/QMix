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

test: clean
	pytest --verbose --color=yes $(TEST_PATH) 

# Clean ----------------------------------------------------------------------

clean_test:
	find . -name 'htmlcov' -exec rm -rf {} +

clean_bytecode:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean_hidden:
	find . -name '.coverage' -exec rm -rf {} +
	find . -name '.cache' -exec rm -rf {} +
	find . -name '.ropeproject' -exec rm -rf {} +
	find . -name '.DS_Store' -exec rm -rf {} +

clean: clean_bytecode clean_hidden clean_test

# Misc -----------------------------------------------------------------------

.PHONY: all test clean_test clean clean_bytecode clean_hidden
