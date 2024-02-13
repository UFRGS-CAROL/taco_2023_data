# Data for the "Assessing the Impact of Compiler Optimizations on GPUs Reliability"

This repository contains the raw data generated for ACM Transactions on Architecture and Code Optimization in 2024.

# Requirements

## To only re-generate the graphs from the paper

- Python >=3.8
- Python pip
- Install all the requirements in the requirements.txt file
  ```shell
  # Install python3-pip
  sudo apt install python3-pip
  # Install python requirements
  python3 -m pip install -r requirements.txt
  ```
  
## To re-do the fault injection and profiling

### NVIDIA tools and boards:

- [NVIDIA Bit Fault Injector](https://github.com/NVlabs/nvbitfi)
- [NVIDIA NVProf](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#) to profile SM 35 kernels.
- [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) to profile SM 70
  kernels.
- An NVIDIA Tesla K40c GPU and an NVIDIA Tesla V100 GPU.

### Applications source code:

In general, we used modified versions of [CUDA samples](https://github.com/NVIDIA/cuda-samples) and part of
[Rodinia](https://www.cs.virginia.edu/rodinia/doku.php?id=start) benchmarks.
We made changes to the applications to conduct fault injection and beam experiments.

Modified code source can be found here:
[BFS](https://github.com/UFRGS-CAROL/radiation-benchmarks/tree/master/src/cuda/bfs),
[LavaMD](https://github.com/UFRGS-CAROL/radiation-benchmarks/blob/master/src/cuda/lava_mp),
[Hotspot](https://github.com/UFRGS-CAROL/radiation-benchmarks/tree/master/src/cuda/hotspot),
[Gaussian](https://github.com/UFRGS-CAROL/radiation-benchmarks/tree/master/src/cuda/gaussian),
[LUD](https://github.com/UFRGS-CAROL/radiation-benchmarks/tree/master/src/cuda/lud),
[CFD](https://github.com/UFRGS-CAROL/radiation-benchmarks/tree/master/src/cuda/cfd), and
[GEMM](https://github.com/UFRGS-CAROL/radiation-benchmarks/tree/master/src/cuda/gemm)

## Data parsing

In the [parsers](parsers/) directory, 
you can find the scripts and data required to regenerate the graphs from the paper.
To regenerate the graphs, run the commands given below:
```shell
cd parsers/
mkdir sheets/
mkdir fig/
cd data/
# Uncompress the files
tar xzf compressed_files.tar.gz 
cd ../
# Run the scripts
./draw_ChipIR_cross_section.py
./draw_error_probability.py
```
The data that has been parsed will be saved in the directories named data/, sheets/, and fig/.

### Plotting the graphs

# Citing this research

Use the following citation:

```bibtex
    @article{10.1145/3638249,
        author = {Santos, Fernando Fernandes dos and Carro, Luigi and Vella, Flavio and Rech, Paolo},
        title = {Assessing the Impact of Compiler Optimizations on GPUs Reliability},
        year = {2024},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        issn = {1544-3566},
        url = {https://doi.org/10.1145/3638249},
        note = {Just Accepted},
        journal = {ACM Transactions on Architecture and Code Optimization},
        month = {jan},
        keywords = {error rate, Graphics Processing Units, reliability, reliability, neutron-induced errors}
    } 
```
