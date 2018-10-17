#!/bin/bash
CDIR=$(pwd)

if [[ ":$PYTHONPATH:" == *":$CDIR:"* ]]; then
  echo "The current directory is already included in the PYTHONPATH environment variable. No action needed."
else
  echo "The current directory is NOT included in the PYTHONPATH environment variable. Adding it now."
  echo " " >> ~/.bash_profile
  echo " " >> ~/.bash_profile
  echo "# Added by the QMix package: " >> ~/.bash_profile
  echo "export PYTHONPATH=\"$(pwd):\$PYTHONPATH\"" >> ~/.bash_profile
  echo "Done."
fi

source ~/.bash_profile
