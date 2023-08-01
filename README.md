# NeuralEstimators

This repository contains the source code for the `R` interface to the `Julia` package `NeuralEstimators`, which facilitates the development of neural estimators in a user-friendly manner. The native `Julia` version is available [here](https://github.com/msainsburydale/NeuralEstimators.jl).

## Getting started

See the package vignette for an overview of the framework and package, and for an illustrative example. You can view the vignette directly in your browser by clicking [here](https://raw.githack.com/msainsburydale/NeuralEstimators/main/NeuralEstimators.html). 

## Installation tips

To install `NeuralEstimators`, please:

1. Install `Julia` (see [here](https://julialang.org/)) and `R` (see [here](https://www.r-project.org/)).
	- Ensure that your system can find the `julia` executable (this usually needs to be done manually; see, e.g., [here](https://julialang.org/downloads/platform/#linux_and_freebsd)) by entering `julia` from the terminal, which should open the Julia REPL (run `exit()` to leave the REPL).
1. Install the `Julia` version of `NeuralEstimators`.
	- To install from terminal, run the command `julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/msainsburydale/NeuralEstimators.jl"))'`.
1. Install the deep-learning library `Flux`.
	- To install from terminal, run the command `julia -e 'using Pkg; Pkg.add("Flux")'`
1. Install the `R` interface to `NeuralEstimators`.
 	- To install from terminal, run the command `Rscript -e 'devtools::install_github("msainsburydale/NeuralEstimators", build_vignettes = TRUE)'`.

Note that the vignette takes roughly 5 minutes to compile; if you are not willing to wait this long, remove the argument `build_vignettes=TRUE` in the final command above.

## Conda

Users may wish to try out `NeuralEstimators` without affecting their current installation. `conda` is a useful tool for this purpose; if you want to run `NeuralEstimators` within a conda environment, you can create one as follows:

```
conda create -n NeuralEstimators -c conda-forge julia r-base nlopt
```

## Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the repository. If you use `NeuralEstimators` in your research or other activities, please use the following citation.

```
@article{SZH_2022_neural_estimators,
	author = {Sainsbury-Dale, Matthew and Zammit-Mangion, Andrew and Huser, Raphaël},
	title = {Likelihood-Free Parameter Estimation with Neural {B}ayes Estimators},
	journal={arXiv:2208.12942},
	year={2022}
}
```


<!-- This package cannot go on CRAN as is, because of the dependence on Julia. The vignette might need to be pre-compiled. I do this in one of my packages, EFDR, where essentially the "vignette" is a link to an HTML file included elsewhere in the package (inst/doc I believe). In your case you may even point it towards the output of the Github Actions CI once you get that working, that would be better. -->
