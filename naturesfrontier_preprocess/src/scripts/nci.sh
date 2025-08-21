#/bin/zsh

cd /Users/peterhawthorne/Projects/WBNCI/wbnci/src/scripts
python preprocess.py $1
cd /Users/peterhawthorne/.julia/dev/NaturalCapitalIndex
/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia run_nci.jl $1
cd /Users/peterhawthorne/Projects/WBNCI/wbnci/src/scripts
python postprocess.py $1
