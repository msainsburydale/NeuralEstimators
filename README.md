# NeuralEstimators <img align="right" width="200" src="https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/docs/src/assets/logo.png?raw=true">

[![][docs-dev-img]][docs-dev-url]
[![R-CMD-check](https://github.com/msainsburydale/NeuralEstimators/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/msainsburydale/NeuralEstimators/actions/workflows/R-CMD-check.yaml)
[![codecov](https://codecov.io/gh/msainsburydale/NeuralEstimators/branch/main/graph/badge.svg?token=6cXItEsKs5)](https://app.codecov.io/gh/msainsburydale/NeuralEstimators)

[docs-dev-img]: https://img.shields.io/badge/vignette-blue.svg
[docs-dev-url]: https://raw.githack.com/msainsburydale/NeuralEstimators/main/inst/doc/NeuralEstimators.html

[julia-repo-img]: https://img.shields.io/badge/Julia_repo-purple.svg
[julia-repo-url]: https://github.com/msainsburydale/NeuralEstimators.jl

[julia-docs-img]: https://img.shields.io/badge/Julia_docs-purple.svg
[julia-docs-url]: https://msainsburydale.github.io/NeuralEstimators.jl/dev/

This repository contains the `R` interface to the `Julia` package `NeuralEstimators` (see [here](https://github.com/msainsburydale/NeuralEstimators.jl)). The package facilitates the user-friendly development of neural point estimators, which are neural networks that map data to a point summary of the posterior distribution. These estimators are likelihood-free and amortised, in the sense that, after an initial setup cost, inference from observed data can be made in a fraction of the time required by conventional approaches. It also facilitates the construction of neural networks that approximate the likelihood-to-evidence ratio in an amortised fashion, which allows for making inference based on the likelihood function or the entire posterior distribution. The package caters for any model for which simulation is feasible by allowing the user to implicitly define their model via simulated data. See the [vignette](https://raw.githack.com/msainsburydale/NeuralEstimators/main/inst/doc/NeuralEstimators.html) to get started!

### Installation

To install the package, please:

1. Install `Julia` (see [here](https://julialang.org/downloads/)) and `R` (see [here](https://www.r-project.org/)).
1. Install the Julia version of `NeuralEstimators`.
	- To install the current stable version of the package from terminal, run the command `julia -e 'using Pkg; Pkg.add("NeuralEstimators")'`. 
	- Alternatively, one may install the development version using `julia -e 'using Pkg; Pkg.add(url="https://github.com/msainsburydale/NeuralEstimators.jl")'`.
1. Install the `R` interface to `NeuralEstimators`.
 	- The package is available on [CRAN](https://CRAN.R-project.org/package=NeuralEstimators), so one may simply run `install.packages("NeuralEstimators")` within `R`. 
 	- Alternatively, one may install the development version by installing `devtools` and running `devtools::install_github("msainsburydale/NeuralEstimators")`. 

Note that if you wish to simulate training data "on-the-fly" using `R` functions, you will also need to install the Julia package `RCall`. Note also that one may compile the vignettes during installation (which takes roughly 10 minutes) by adding the argument `build_vignettes = TRUE` in the final command above.   

### Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the repository. If you use it in your research or other activities, please also use the following citation.

```
@article{,
	author = {Sainsbury-Dale, Matthew and Zammit-Mangion, Andrew and Huser, Raphaël},
	title = {Likelihood-Free Parameter Estimation with Neural {B}ayes Estimators},
	journal = {The American Statistician},
	year = {2024},
	volume = {78},
	pages = {1--14},
	doi = {10.1080/00031305.2023.2249522},
	url = {https://doi.org/10.1080/00031305.2023.2249522}
}
```

### Papers using NeuralEstimators

- **Likelihood-free parameter estimation with neural Bayes estimators** [[paper]](https://doi.org/10.1080/00031305.2023.2249522) [[code]](https://github.com/msainsburydale/NeuralBayesEstimators)

- **Neural Bayes estimators for censored inference with peaks-over-threshold models** [[paper]](https://arxiv.org/abs/2306.15642)

- **Neural Bayes estimators for irregular spatial data using graph neural networks** [[paper]](https://arxiv.org/abs/2310.02600)[[code]](https://github.com/msainsburydale/NeuralEstimatorsGNN)

- **Modern extreme value statistics for Utopian extremes** [[paper]](https://arxiv.org/abs/2311.11054)

- **Neural Methods for Amortised Inference** [[paper]](https://arxiv.org/abs/2404.12484)[[code]](https://github.com/andrewzm/Amortised_Neural_Inference_Review)


### Related packages 

Several other software packages have been developed to facilitate neural likelihood-free inference. These include:

- [BayesFlow](https://github.com/bayesflow-org/bayesflow) (TensorFlow)
- [LAMPE](https://github.com/probabilists/lampe) (PyTorch)
- [sbi](https://github.com/sbi-dev/sbi) (PyTorch)
- [swyft](https://github.com/undark-lab/swyft) (PyTorch)

A summary of the functionality in these packages is given in [Zammit-Mangion et al. (2024, Section 6.1)](https://arxiv.org/abs/2404.12484). Note that this list of related packages was created in July 2024; if you have software to add to this list, please contact the package maintainer. 
