IMAGENAME = dcv-benchmarks

BENCH_SCRIPT = /work/bench_async_search.py

build:
	docker build -t $(IMAGENAME) .

bash: build
	docker run -it --cpus=$(CPU_COUNT) --workdir /work/ -v $(shell pwd)/examples/output_data:/data $(IMAGENAME) /bin/bash

bench: build
	docker run -it --cpus=$(CPU_COUNT) --workdir /work/ -v $(shell pwd)/examples/output_data:/data $(IMAGENAME) python $(BENCH_SCRIPT) $(ARGS)
