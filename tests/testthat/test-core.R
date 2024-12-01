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
  installed = "Flux" ∈ keys(Pkg.project().dependencies)
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
  installed = "NeuralEstimators" ∈ keys(Pkg.project().dependencies)
  if !installed
    Pkg.add(url = "https://github.com/msainsburydale/NeuralEstimators.jl") 
  end
  using NeuralEstimators
')
  expect_equal(1, 1)
})

test_that("Optim.jl is available", {
  juliaEval('
  # Install the package if not already installed
  using Pkg
  installed = "Optim" ∈ keys(Pkg.project().dependencies)
  if !installed
    Pkg.add("Optim") 
  end
  using Optim
')
  expect_equal(1, 1)
})

test_that("a neural estimator can be initialised", {
  
  ## Using Flux code directly
  estimator <<- juliaEval('
  using NeuralEstimators, Flux

  p = 2    # number of parameters in the statistical model
  w = 32   # number of neurons in each layer

  psi = Chain(Dense(1, w, relu), Dense(w, w, relu), Dense(w, w, relu))
  phi = Chain(Dense(w, w, relu), Dense(w, p))
  estimator = PointEstimator(DeepSet(psi, phi))
')
  
  ## Using the helper function
  p = 2
  initialise_estimator(p, architecture = "MLP")
  initialise_estimator(p, architecture = "MLP", variance_stabiliser = 'x -> cbrt.(x)')
  initialise_estimator(p, architecture = "GNN")
  initialise_estimator(p, architecture = "CNN", kernel_size = list(10, 5, 3))
  initialise_estimator(p, architecture = "CNN", kernel_size = list(c(10, 10), c(5, 5), c(3, 3)))
  expect_error(initialise_estimator(p, architecture = "CNN"))
  
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
  
  expect_error(train(estimator, Z_train = Z_train, Z_val = Z_val, epochs = 2, verbose = F))
  
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

# NB This requires the user to have installed the Julia package RCall
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
      	mu = randn(K)
      	sigma = rand(K)
      	theta = hcat(mu, sigma)'
      	return theta
      end")
  
  # Data simulator
  simulator <- juliaEval("
      simulator(theta_matrix, m) = [theta[1] .+ theta[2] * randn(1, m) for theta in eachcol(theta_matrix)]
      ")
  
  
  estimator  <- train(estimator, sampler = sampler, simulator = simulator, m = m, epochs = 2, verbose = F)
  estimator  <- train(estimator, sampler = sampler, simulator = simulator, m = m, epochs = 2, loss = "squared-error", verbose = F)
  estimator  <- train(estimator, sampler = sampler, simulator = simulator, m = m, epochs = 2, loss = "Flux.Losses.mae", verbose = F)
  estimators <- train(estimator, sampler = sampler, simulator = simulator, m = c(m, 2*m), epochs = 2, verbose = F)
  
  expect_warning(train(estimator, sampler = sampler, simulator = simulator, M = m, epochs = 2, verbose = F))
  expect_error(train(estimator, sampler = sampler, simulator = simulator))
  expect_error(train(estimator, sampler = sampler, Z_train = Z_train, Z_val = Z_val, epochs = 2, verbose = F))
  expect_error(train(estimator, sampler = sampler, theta_train = theta_train, theta_val = theta_val, epochs = 2, verbose = F))
  expect_error(train(estimator, sampler = sampler, epochs = 2, verbose = F))
  expect_error(train(estimator, theta_train = theta_train, theta_val = theta_val, simulator = simulator, Z_train = Z_train, Z_val = Z_val, epochs = 2, verbose = F))
  
  expect_equal(1, 1)
})

test_that("the neural estimator can be assessed with assess()", {
  theta_test  <- sampler(100)
  Z_test      <- simulator(theta_test, m)
  assessment  <- assess(estimator, theta_test, Z_test)
  risk(assessment)
  risk(assessment$estimates)
  bias(assessment)
  bias(assessment$estimates)
  rmse(assessment)
  rmse(assessment$estimates)
  
  # Test that parameters can be given as a vector in single-parameter case
  estimator_one_param <- initialise_estimator(1, architecture = "MLP")
  assessment <- assess(estimator_one_param, rnorm(100), Z_test)
  
  expect_equal(1, 1)
})

test_that("the neural estimator can be applied to real data using estimate() and bootstrap()", {
  # Generate some "observed" data
  theta    <- as.matrix(c(0, 0.5))         # true parameters
  Z        <- simulator(theta, m)          # pretend that this is observed data
  thetahat <- estimate(estimator, Z)       # point estimates
  p = 2
  expect_equal(nrow(thetahat), p)
  expect_equal(ncol(thetahat), 1)
  
  ## Non-parametric bootstrap estimates
  B  <- 400
  bs <- bootstrap(estimator, Z, B = B)  
  expect_equal(nrow(bs), p)
  expect_equal(ncol(bs), B)
})

test_that("neural ratio estimator can be constructed and used to make inference", {
  
  estimator <- juliaEval('
    using NeuralEstimators, Flux
    p = 2    # number of parameters in the statistical model
    w = 32   # number of neurons in each layer
    psi = Chain(Dense(1, w, relu), Dense(w, w, relu), Dense(w, w, relu))
    phi = Chain(Dense(w+p, w, relu), Dense(w, 1))
    deepset = DeepSet(psi, phi)
    estimator = RatioEstimator(deepset)
')
  
  theta <- as.matrix(c(0, 0.5))         # true parameters
  Z     <- simulator(theta, m)          # "observed" data
  ratio <- estimate(estimator, Z, theta)   # ratio estimate
  expect_equal(nrow(ratio), 1)
  expect_equal(ncol(ratio), 1)
  ratio <- estimate(estimator, Z, cbind(theta, theta))   # ratio estimates 
  expect_equal(nrow(ratio), 1)
  expect_equal(ncol(ratio), 2)
  expect_error(estimate(estimator, list(Z, Z), theta))
  
  # Grid-based methods for estimation/posterior sampling
  theta_grid <- t(expand.grid(seq(0, 1, len = 50), seq(0, 1, len = 50)))
  mlestimate(estimator, Z, theta_grid = theta_grid)
  mapestimate(estimator, Z, theta_grid = theta_grid)
  # mapestimate(estimator, Z, theta_grid = theta_grid, prior = function(x) 1) # NB This requires the user to have installed the Julia package RCall, which is not particularly stable
  sampleposterior(estimator, Z[[1]], theta_grid = theta_grid) 
  sampleposterior(estimator, Z, theta_grid = theta_grid)
  expect_error(sampleposterior(estimator, c(Z, Z), theta_grid = theta_grid))
  # sampleposterior(estimator, Z, theta_grid = theta_grid, prior = function(x) 1) # NB This requires the user to have installed the Julia package RCall
  
  # Gradient descent method for estimation
  # juliaEval('using Optim')
  # theta0 <- c(0.2, 0.4)
  # mlestimate(estimator, Z, theta0 = theta0) 
  # mapestimate(estimator, Z, theta0 = theta0) 
})
