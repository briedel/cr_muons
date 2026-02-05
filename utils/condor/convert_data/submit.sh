#!/bin/bash 


# eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)

# /home/briedel/code/cr_muons/icetray/build/bin/icetray-shell 


range=$(dirname $1 | awk -F/ '{print $(NF-1)}')

out=/data/sim/IceCube/2025/testing/$range/$(basename $1).parquet

echo $range
echo $out

if [[ -f "$out" ]]; then
    echo "File $out exists. Exit 0"
    exit 0
fi 

/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/icetray-env icetray/v1.17.0 python3 /home/briedel/code/cr_muons_git/utils/data_converters/dump_muonitron_data.py --infile $1 --outfile $out  --format parquet
