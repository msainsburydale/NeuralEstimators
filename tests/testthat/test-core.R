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

test_that("Lux.jl is available", {
  juliaEval('
  using Pkg
  installed = "Lux" in keys(Pkg.project().dependencies)
  if !installed
    Pkg.add("Lux")
  end
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

test_that("Optim.jl is available", {
  juliaEval('
  # Install the package if not already installed
  using Pkg
  installed = "Optim" in keys(Pkg.project().dependencies)
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

  psi = Flux.Chain(Flux.Dense(1, w, relu), Flux.Dense(w, w, relu), Flux.Dense(w, w, relu))
  phi = Flux.Chain(Flux.Dense(w, w, relu), Flux.Dense(w, p))
  estimator = PointEstimator(DeepSet(psi, phi))
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
  estimator  <- train(estimator, sampler = sampler, simulator = simulator, m = m, epochs = 2, verbose = F,
                      device = cpu_device(), epochs_per_refresh = 2)
  
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
  bias(assessment)
  rmse(assessment)
  
  # Test that parameters can be given as a vector in single-parameter case
  estimator_one_param <- juliaEval('
    using NeuralEstimators, Flux
    p = 1    # number of parameters in the statistical model
    w = 32   # number of neurons in each layer
    psi = Flux.Chain(Flux.Dense(1, w, relu), Flux.Dense(w, w, relu), Flux.Dense(w, w, relu))
    phi = Flux.Chain(Flux.Dense(w, w, relu), Flux.Dense(w, p))
    estimator = PointEstimator(DeepSet(psi, phi))
  ')
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
  thetahat_infer <- infer(estimator, Z)
  expect_equal(dim(thetahat_infer), dim(thetahat))
  
  ## Non-parametric bootstrap estimates
  B  <- 400
  bs <- bootstrap(estimator, Z, B = B)  
  expect_equal(nrow(bs), p)
  expect_equal(ncol(bs), B)
})

test_that("R wrappers construct PointEstimator and RatioEstimator", {

  network <- juliaEval('
    using NeuralEstimators, Flux
    w = 32
    psi = Flux.Chain(Flux.Dense(1, w, relu), Flux.Dense(w, w, relu))
    phi = Flux.Chain(Flux.Dense(w, w, relu), Flux.Dense(w, 2))
    DeepSet(psi, phi)
  ')
  point <- PointEstimator(network)
  theta_train <- sampler(50)
  theta_val   <- sampler(50)
  Z_train <- simulator(theta_train, m)
  Z_val   <- simulator(theta_val, m)
  point <- train(point, theta_train = theta_train, theta_val = theta_val,
                 Z_train = Z_train, Z_val = Z_val, epochs = 2, verbose = FALSE)
  Z <- simulator(as.matrix(c(0, 0.5)), m)
  thetahat <- infer(point, Z)
  expect_equal(nrow(thetahat), 2)
  expect_equal(ncol(thetahat), 1)

  summary_network <- juliaEval('
    using NeuralEstimators, Flux
    d = 2
    w = 32
    num_summaries = 3d
    psi = Flux.Chain(Flux.Dense(1, w, relu), Flux.Dense(w, w, relu))
    phi = Flux.Chain(Flux.Dense(w, w, relu), Flux.Dense(w, num_summaries))
    DeepSet(psi, phi)
  ')
  ratio <- RatioEstimator(2, summary_network, num_summaries = 6)
  Z <- simulator(as.matrix(c(0, 0.5)), m)
  r <- logratio(ratio, Z, as.matrix(c(0, 0.5)))
  expect_equal(nrow(r), 1)
  expect_equal(ncol(r), 1)
})

test_that("neural ratio estimator can be constructed and used to make inference", {
  
  estimator <- juliaEval('
    using NeuralEstimators, Flux
    d = 2    # number of parameters in the statistical model
    w = 32   # number of neurons in each layer
    num_summaries = 3d
    psi = Flux.Chain(Flux.Dense(1, w, relu), Flux.Dense(w, w, relu), Flux.Dense(w, w, relu))
    phi = Flux.Chain(Flux.Dense(w, w, relu), Flux.Dense(w, num_summaries))
    summary_network = DeepSet(psi, phi)
    estimator = RatioEstimator(summary_network, d; num_summaries = num_summaries)
')
  
  theta <- as.matrix(c(0, 0.5))            # true parameters
  Z     <- simulator(theta, m)             # "observed" data
  ratio <- logratio(estimator, Z, theta)   # ratio estimate
  expect_equal(nrow(ratio), 1)
  expect_equal(ncol(ratio), 1)
  ratio <- logratio(estimator, Z, cbind(theta, theta))   # ratio estimates 
  expect_equal(nrow(ratio), 1)
  expect_equal(ncol(ratio), 2)
  
  # Grid-based methods for estimation/posterior sampling
  grid <- t(expand.grid(seq(0, 1, len = 50), seq(0, 1, len = 50)))
  samples <- sampleposterior(estimator, Z[[1]], grid = grid, N = 50)
  expect_equal(length(dim(samples)), 3)
  samples <- sampleposterior(estimator, Z, grid = grid, N = 50)
  expect_equal(length(dim(samples)), 3)
  samples_infer <- infer(estimator, Z, grid = grid, N = 50)
  expect_equal(length(dim(samples_infer)), 3)
})

test_that("a Lux estimator can be trained, saved, loaded, and used for inference", {

  lux_estimator <- juliaEval('
    using NeuralEstimators, Lux
    d = 2
    n = 15
    network = MLP(n, d; depth = 2, width = 32, backend = Lux)
    PointEstimator(network)
  ')

  theta_train <- sampler(100)
  theta_val   <- sampler(100)
  # MLP maps an n-vector; store each data set as a length-m vector (matrix with m rows)
  lux_simulator <- function(theta_set, m) {
    apply(theta_set, 2, function(theta) rnorm(m, theta[1], theta[2]))
  }
  Z_train <- lux_simulator(theta_train, m)
  Z_val   <- lux_simulator(theta_val, m)

  lux_estimator <- train(
    lux_estimator,
    theta_train = theta_train,
    theta_val   = theta_val,
    Z_train = Z_train,
    Z_val   = Z_val,
    epochs = 2,
    verbose = FALSE,
    device = cpu_device()
  )

  Z <- matrix(lux_simulator(as.matrix(c(0, 0.5)), m), nrow = m)
  thetahat <- estimate(lux_estimator, Z)
  expect_equal(nrow(thetahat), 2)
  expect_equal(ncol(thetahat), 1)
  thetahat_infer <- infer(lux_estimator, Z)
  expect_equal(dim(thetahat_infer), dim(thetahat))

  filename <- tempfile(fileext = ".bson")
  savestate(lux_estimator, filename)
  lux_reload <- juliaEval('
    using NeuralEstimators, Lux
    d = 2
    n = 15
    network = MLP(n, d; depth = 2, width = 32, backend = Lux)
    PointEstimator(network)
  ')
  lux_reload <- loadstate(lux_reload, filename)
  thetahat_reload <- estimate(lux_reload, Z)
  expect_equal(thetahat_reload, thetahat)
})
