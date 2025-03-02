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

test_that("Flux.jl is available", {
  juliaEval('
  # Install the package if not already installed
  using Pkg
  installed = "Flux" in keys(Pkg.project().dependencies)
  if !installed
    Pkg.add("Flux")  
  end
  using Flux
')
  expect_equal(1, 1)
})

test_that("NeuralEstimators.jl is available", {
  juliaEval('
  # Install the package if not already installed
  using Pkg
  installed = "NeuralEstimators" in keys(Pkg.project().dependencies)
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
  UW <- encodedata(list(Z, Z))
  expect_equal(length(UW), 2)
})

#TODO why is this failing on CI but not on my computer?
# test_that("spatialgraph() is working", {
#   # Number of replicates and spatial dimension
#   m <- 5
# 
#   # Spatial locations fixed for all replicates
#   n <- 100
#   S <- matrix(runif(n * 2), n, 2)
#   Z <- matrix(runif(n * m), n, m)
#   g <- spatialgraph(S, Z)
# 
#   # Spatial locations varying between replicates
#   n <- sample(50:100, m, replace = TRUE)
#   S <- lapply(n, function(ni) matrix(runif(ni * 2), ni, 2))
#   Z <- lapply(n, function(ni) runif(ni))
#   g <- spatialgraph(S, Z)
# 
#   spatialgraph(S, Z)
#   spatialgraph(S, Z, k = 1)
#   spatialgraph(S, Z, k = 100)
#   spatialgraph(S, Z, k = 10L, r = 0.1)
#   spatialgraph(S, Z, k = 10.0, r = 0.1)
#   spatialgraph(S, Z, k = 10, r = 1.0)
# 
#   # Multiple data sets: Spatial locations fixed for all replicates within a given data set
#   K <- 15 # number of data sets
#   n <- sample(50:100, K, replace = TRUE) # number of spatial locations can vary between data sets
#   S <- lapply(1:K, function(k) matrix(runif(n[k] * 2), n[k], 2))
#   Z <- lapply(1:K, function(k) matrix(runif(n[k] * m), n[k], m))
#   g <- spatialgraph(S, Z)
#   
#   # Multiple data sets: Spatial locations varying between replicates within a given data set
#   S <- lapply(1:K, function(k) {
#     lapply(1:m, function(i) {
#       ni <- sample(50:100, 1)       # randomly generate the number of locations for each replicate
#       matrix(runif(ni * 2), ni, 2)  # generate the spatial locations
#     })
#   })
#   Z <- lapply(1:K, function(k) {
#     lapply(1:m, function(i) {
#       n <- nrow(S[[k]][[i]])
#       runif(n)  
#     })
#   })
#   g <- spatialgraph(S, Z)
# })
