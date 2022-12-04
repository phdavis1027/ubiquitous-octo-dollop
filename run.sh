#!/bin/bash

if [[ $# -eq 0 ]]; then
  python to_jl.py
  python csv_maker.py
  python main.py
elif [[ $1 -eq "data" ]]; then
  python to_jl.py
  python csv_maker.py
elif [[ $1 -eq "learn" ]]; then
  python main.py
fi
