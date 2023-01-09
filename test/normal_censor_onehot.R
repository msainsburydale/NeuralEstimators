library("NeuralEstimators")
library("JuliaConnectoR")

prior <- function(K) {
  mu    <- rnorm(K)
  sigma <- rgamma(K, 1)
  theta <- matrix(c(mu, sigma), byrow = TRUE, ncol = K)
  return(theta)
}
theta_train = prior(10000)
theta_val   = prior(2000)
theta_test  = prior(1000)


#Varying threshold
censor.threshold.p<-0.8
simulate <- function(theta_set, m) {
  apply(theta_set, 2, function(theta) {
    censor.threshold <- qnorm(censor.threshold.p,theta[1],theta[2])
    Z <- rnorm(m, theta[1], theta[2])
    #Z[Z<censor.threshold]<-NA
    Z[Z <= censor.threshold]<- censor.threshold

    Z<-rbind(Z,1*(Z > censor.threshold))
    dim(Z) <- c(2, 1, m)
    Z
  }, simplify = FALSE)
}

m <- 200
set.seed(1)
Z_train <- simulate(theta_train, m)
Z_val   <- simulate(theta_val, m)

estimator1 <- juliaEval('

  n = 2    # size of each replicate (univariate data)
  w = 32   # number of neurons in each layer
  p = 2    # number of parameters in the statistical model

  using Flux
  psi = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
  phi = Chain(Dense(w, w, relu), Dense(w, w, relu), Dense(w, p), Flux.flatten)

  using NeuralEstimators
  estimator = DeepSet(psi, phi)
')


estimator1.trained <- train(
  estimator1,
  theta_train = theta_train,
  theta_val   = theta_val,
  Z_train = Z_train,
  Z_val   = Z_val,
  epochs = 100L,
  batchsize=8L,
  use_gpu = F
)


J     <- 300
theta <- as.matrix(c(0, 1))
Z     <- lapply(1:J, function(i) simulate(theta, m))
Z     <- do.call(c, Z)
assessment <- assess(list(estimator1.trained), theta, list(Z))

plotdistribution(assessment$estimates, type = "scatter")

estimator1.trained <- juliaLet('cpu(estimator)', estimator = estimator1.trained) # move the estimator to the cpu

Z <- simulate(theta, m) # pretend that this is observed data
estimate(estimator1.trained, Z)

