Benchmarking dask-searchcv with docker. 

Useful for running benchmarks multiple times in a (semi?-)isolated fashion on spot instances.

Run with:

`make build`

`make bench CPU_COUNT=4 ARGS=--lparam=4 --version=3`

`python run_bench.py` - runs the above many times over a grid of parameters
