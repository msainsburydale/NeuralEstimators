testthat::skip_on_cran()

set.seed(1)

test_that("packages can be loaded properly", {
  library("NeuralEstimators")
  library("JuliaConnectoR")
  expect_equal(1, 1)
})

test_that("julia can be called", {
  x <- juliaEval('
  1 + 1
')
  expect_equal(x, 2)
})

test_that("Flux is available", {
  juliaEval('
  # Install the package if not already installed
  using Pkg
  installed = "Flux" ∈ keys(Pkg.project().dependencies)
  if !installed
    Pkg.add("Flux")  
  end
  using Flux
')
  expect_equal(1, 1)
})

test_that("the Julia version of NeuralEstimators is available", {
  juliaEval('
  # Install the package if not already installed
  using Pkg
  installed = "NeuralEstimators" ∈ keys(Pkg.project().dependencies)
  if !installed
    Pkg.add(url = "https://github.com/msainsburydale/NeuralEstimators.jl") 
  end
  using NeuralEstimators
')
  expect_equal(1, 1)
})

test_that("encodedata() is working", {
  Z <- matrix(c(1, 2, NA, NA, 5, 6, 7, NA, 9), nrow = 3)
  UW <- encodedata(Z)
  expect_equal(dim(UW), c(3, 3, 2, 1))
  UW <- encodedata(list(Z, Z))
  expect_equal(length(UW), 2)
  expect_equal(dim(UW[[1]]), c(3, 3, 2, 1))
})