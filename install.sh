#!/bin/bash
CDIR=$(pwd)

if [[ ":$PYTHONPATH:" == *":$CDIR:"* ]]; then
  echo ""
  echo "The current directory is already included in the"
  echo "PYTHONPATH environment variable. No action needed."
  echo ""
else
  echo ""
  echo "The current directory is NOT included in the"
  echo "PYTHONPATH environment variable. Adding now..."
  echo " " >> ~/.bash_profile
  echo " " >> ~/.bash_profile
  echo "# Added by the QMix package: " >> ~/.bash_profile
  echo "export PYTHONPATH=\"$(pwd):\$PYTHONPATH\"" >> ~/.bash_profile
  echo "Done."
  echo ""
fi

source ~/.bash_profile
