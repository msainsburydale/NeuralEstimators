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

test_that("the Julia version of NeuralEstimators is available", {
  juliaEval('
  using NeuralEstimators
')
  expect_equal(1, 1)
})

test_that("Flux is available", {
  juliaEval('
  using Flux
')
  expect_equal(1, 1)
})

test_that("the neural estimator can be initialised", {
  estimator <<- juliaEval('
  using NeuralEstimators
  using Flux

  p = 2    # number of parameters in the statistical model
  w = 32   # number of neurons in each layer

  psi = Chain(Dense(1, w, relu), Dense(w, w, relu), Dense(w, w, relu))
  phi = Chain(Dense(w, w, relu), Dense(w, p))
  estimator = DeepSet(psi, phi)
')
  expect_equal(1, 1)
})

# Sampler from the prior
sampler <- function(K) {
  mu    <- rnorm(K)
  sigma <- rgamma(K, 1)
  theta <- matrix(c(mu, sigma), byrow = TRUE, ncol = K)
  return(theta)
}

# Data simulator
simulator <- function(theta_set, m) {
  apply(theta_set, 2, function(theta) {
    Z <- rnorm(m, theta[1], theta[2])
    dim(Z) <- c(1, m)
    Z
  }, simplify = FALSE)
}
m <- 15


test_that("the neural estimator can be trained with fixed a training set", {
  
  theta_train <- sampler(100)
  theta_val   <- sampler(100)
  Z_train <- simulator(theta_train, m)
  Z_val   <- simulator(theta_val, m)
  
  estimator <- train(
    estimator,
    theta_train = theta_train,
    theta_val   = theta_val,
    Z_train = Z_train,
    Z_val   = Z_val,
    epochs = 2, 
    verbose = F
  )
  expect_equal(1, 1)
})

# NB This requires the user to have installed the Julia package RCall - not sure that I want NeuralEstimators to depend on RCall, so I've omitted this test for now
# test_that("the neural estimator can be trained with simulation on-the-fly (using R functions)", {
#   
#   estimator <- train(
#     estimator,
#     sampler = sampler, 
#     simulator = simulator,
#     m = m, 
#     epochs = 2, 
#     verbose = F
#   )
#   expect_equal(1, 1)
# })

test_that("the neural estimator can be trained with simulation on-the-fly (using Julia functions)", {
  
  # Parameter sampler
  sampler <- juliaEval("
      function sampler(K)
      	μ = randn(K)
      	σ = rand(K)
      	θ = hcat(μ, σ)'
      	return θ
      end")
  
  # Data simulator
  simulator <- juliaEval("
      simulator(θ_matrix, m) = [θ[1] .+ θ[2] * randn(1, m) for θ ∈ eachcol(θ_matrix)]
      ")
  
  estimator <- train(
    estimator,
    sampler = sampler, 
    simulator = simulator,
    m = m, 
    epochs = 2, 
    verbose = F
  )
  expect_equal(1, 1)
})

test_that("the neural estimator can be assessed", {
  theta_test  <- sampler(100)
  Z_test      <- simulator(theta_test, m)
  assessment  <- assess(estimator, theta_test, Z_test)
  expect_equal(1, 1)
})

test_that("the neural estimator can be applied to real data", {
  # Generate some "observed" data
  theta    <- as.matrix(c(0, 0.5))         # true parameters
  Z        <- simulator(theta, m)          # pretend that this is observed data
  thetahat <- estimate(estimator, Z)       # point estimates
  p = 2
  expect_equal(nrow(thetahat), p)
  expect_equal(ncol(thetahat), 1)
  B = 400
  bs <- bootstrap(estimator, Z, B = B)  # non-parametric bootstrap estimates
  expect_equal(nrow(bs), p)
  expect_equal(ncol(bs), B)
})
