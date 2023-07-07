## Test the core functionality of the package using the univariate Gaussian example
test_that("packages can be loaded properly", {
  library("NeuralEstimators")
  library("JuliaConnectoR")
  expect_equal(1, 1)
})

prior <- function(K) {
  mu    <- rnorm(K)
  sigma <- rgamma(K, 1)
  theta <- matrix(c(mu, sigma), byrow = TRUE, ncol = K)
  return(theta)
}
set.seed(1)
theta_train = prior(100)
theta_val   = prior(100)
theta_test  = prior(100)


simulate <- function(theta_set, m) {
  apply(theta_set, 2, function(theta) {
    Z <- rnorm(m, theta[1], theta[2])
    dim(Z) <- c(1, m)
    Z
  }, simplify = FALSE)
}

m <- 15
Z_train <- simulate(theta_train, m)
Z_val   <- simulate(theta_val, m)


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

test_that("the neural estimator can be trained", {
  estimator <- train(
    estimator,
    theta_train = theta_train,
    theta_val   = theta_val,
    Z_train = Z_train,
    Z_val   = Z_val,
    epochs = 2
  )
  expect_equal(1, 1)
})

test_that("the neural estimator can be assessed", {
  theta_test  <- prior(100)
  Z_test      <- simulate(theta_test, m)
  assessment  <- assess(estimator, theta_test, Z_test)
  expect_equal(1, 1)
})

test_that("the neural estimator can be applied to real data", {
  # Generate some "observed" data
  theta    <- as.matrix(c(0, 0.5))         # true parameters
  Z        <- simulate(theta, m)           # pretend that this is observed data
  thetahat <- estimate(estimator, Z)       # point estimates
  p = 2
  expect_equal(nrow(thetahat), p)
  expect_equal(ncol(thetahat), 1)
  B = 400
  bs <- bootstrap(estimator, Z, B = B)  # non-parametric bootstrap estimates
  expect_equal(nrow(bs), p)
  expect_equal(ncol(bs), B)
})



