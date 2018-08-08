# Makefile for QMix
#
# Makefile examples that I drew from:
# https://github.com/pjz/mhi/blob/master/Makefile
# http://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects.html
#

TEST_PATH=./tests/
QMIX_PATH=./qmix/


all: test open_test clean_bytecode clean_hidden


# Add Qmix/ to PYTHONPATH ----------------------------------------------------

init:
	chmod 700 misc/add_pypath.sh
	sh misc/add_pypath.sh


# Install requirements (with Pip) --------------------------------------------

pip_install:
	pip install -r misc/requirements.txt

pip_upgrade:
	pip install -r misc/requirements.txt --upgrade

pip_update: pip_upgrade


# Install requirements (with Conda) ------------------------------------------

conda_install:
	conda install --yes --file=misc/requirements.txt

conda_upgrade:
	conda upgrade --yes --file=misc/requirements.txt

conda_update: conda_upgrade


# Run tests (with Py.Test) ---------------------------------------------------

test: clean
	pytest --verbose --color=yes --cov=$(QMIX_PATH) --cov-report=html $(TEST_PATH) 

open_test:
	open htmlcov/index.html

clean_test:
	find . -name 'htmlcov' -exec rm -rf {} +


# Clean ----------------------------------------------------------------------

clean_bytecode:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean_hidden:
	find . -name '.coverage' -exec rm -rf {} +
	find . -name '.cache' -exec rm -rf {} +
	find . -name '.ropeproject' -exec rm -rf {} +
	find . -name '.DS_Store' -exec rm -rf {} +

clean_all: clean_bytecode clean_hidden clean_test


# Enforce PEP-008 ------------------------------------------------------------

pep8:
	autopep8 qmix --recursive --in-place --pep8-passes 2000 --verbose


# Pylint ---------------------------------------------------------------------

pylint:
	pylint qmix/ --rcfile=misc/pylint.rc


# Misc -----------------------------------------------------------------------

.PHONY: all init pip_install pip_upgrade pip_update conda_install conda_upgrade conda_update clean open_test clean_test clean
