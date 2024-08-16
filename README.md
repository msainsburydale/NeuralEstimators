# NeuralEstimators <img align="right" width="200" src="https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/docs/src/assets/logo.png?raw=true">

[![][docs-dev-img]][docs-dev-url]
[![R-CMD-check](https://github.com/msainsburydale/NeuralEstimators/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/msainsburydale/NeuralEstimators/actions/workflows/R-CMD-check.yaml)
[![codecov](https://codecov.io/gh/msainsburydale/NeuralEstimators/branch/main/graph/badge.svg?token=6cXItEsKs5)](https://codecov.io/gh/msainsburydale/NeuralEstimators)

[docs-dev-img]: https://img.shields.io/badge/vignette-blue.svg
[docs-dev-url]: https://raw.githack.com/msainsburydale/NeuralEstimators/main/NeuralEstimators.html

[julia-repo-img]: https://img.shields.io/badge/Julia_repo-purple.svg
[julia-repo-url]: https://github.com/msainsburydale/NeuralEstimators.jl

[julia-docs-img]: https://img.shields.io/badge/Julia_docs-purple.svg
[julia-docs-url]: https://msainsburydale.github.io/NeuralEstimators.jl/dev/

This repository contains the `R` interface to the `Julia` package `NeuralEstimators` (see [here](https://github.com/msainsburydale/NeuralEstimators.jl)). The package facilitates the user-friendly development of neural point estimators, which are neural networks that transform data into parameter point estimates. They are likelihood free, substantially faster than classical methods, and can be designed to be approximate Bayes estimators. The package caters for any model for which simulation is feasible.  See the [vignette](https://raw.githack.com/msainsburydale/NeuralEstimators/main/NeuralEstimators.html) to get started!

### Installation tips

To install the package, please:

1. Install `Julia` (see [here](https://julialang.org/downloads/)) and `R` (see [here](https://www.r-project.org/)).
1. Install the Julia version of `NeuralEstimators`.
	- To install from terminal, run the command `julia -e 'using Pkg; Pkg.add(url="https://github.com/msainsburydale/NeuralEstimators.jl")'`.
1. Install the `R` interface to `NeuralEstimators`.
 	- Install and load `devtools` in R and then run `devtools::install_github("msainsburydale/NeuralEstimators")`.

Note that if you wish to simulate training data "on-the-fly" using `R` functions, you will also need to install the Julia package `RCall`. Note also that one may compile the vignette during installation (which takes roughly 5 minutes) by adding the argument `build_vignettes = TRUE` in the final command above.   

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

- **Likelihood-free parameter estimation with neural Bayes estimators** [[paper]](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522) [[code]](https://github.com/msainsburydale/NeuralBayesEstimators)

- **Neural Bayes estimators for censored inference with peaks-over-threshold models** [[paper]](https://arxiv.org/abs/2306.15642)

- **Neural Bayes estimators for irregular spatial data using graph neural networks** [[paper]](https://arxiv.org/abs/2310.02600)[[code]](https://github.com/msainsburydale/NeuralEstimatorsGNN)

- **Modern extreme value statistics for Utopian extremes** [[paper]](https://arxiv.org/abs/2311.11054)

- **Neural Methods for Amortised Inference** [[paper]](https://arxiv.org/abs/2404.12484)[[code]](https://github.com/andrewzm/Amortised_Neural_Inference_Review)


### Related packages 

Several other software packages have been developed to facilitate neural likelihood-free inference. These include:

1. [BayesFlow](https://github.com/stefanradev93/BayesFlow) (TensorFlow)
1. [LAMPE](https://github.com/probabilists/lampe) (PyTorch)
1. [sbi](https://github.com/sbi-dev/sbi) (PyTorch)
1. [swyft](https://github.com/undark-lab/swyft) (PyTorch)


A summary of the functionality in these packages is given in [Zammit-Mangion et al. (2024, Section 6.1)](https://arxiv.org/abs/2404.12484). Note that this list of related packages was created in July 2024; if you have software to add to this list, please contact the package maintainer. 

<!-- This package cannot go on CRAN as is, because of the dependence on Julia. The vignette might need to be pre-compiled. I do this in one of my packages, EFDR, where essentially the "vignette" is a link to an HTML file included elsewhere in the package (inst/doc I believe). In your case you may even point it towards the output of the Github Actions CI once you get that working, that would be better. -->
