# Wrapper functions around the core functions in NeuralEstimators.jl.

#TODO simulation on the fly
#TODO custom loss functions
#TODO optimiser

#' @title train a neural estimator
#' @export
train <- function(estimator,
                  theta_train,
                  theta_val,
                  Z_train,
                  Z_val,
                  M = NULL, # if M is left NULL, it is ignored
                  #loss = Flux.Losses.mae, #TODO need to be able to provide a julia function so that we can cater for arbitrary loss functions
                  #optimiser = ADAM(1e-4), #TODO maybe just ask the user for the learning rate? Or allow the user to provide either a single number (the learning rate) or a julia object.
                  epochs = 100L,
                  batchsize = 32L,
                  savepath = "",
                  stopping_epochs = 10L,
                  use_gpu = TRUE,
                  verbose = TRUE
                  ) {



  estimator = juliaLet('

  using NeuralEstimators
  using Flux

  # Convert parameters and data to Float32 for computational efficiency
  theta_train = broadcast.(Float32, theta_train)
  theta_val = broadcast.(Float32, theta_val)
  Z_train = broadcast.(Float32, Z_train)
  Z_val   = broadcast.(Float32, Z_val)


  # TODO too much repetition here.
  if isnothing(M)
      estimator = train(
        estimator, theta_train, theta_val, Z_train, Z_val,
        # loss = loss, optimiser = optimiser, #TODO uncomment when I figure it out
        epochs = epochs, batchsize = batchsize,
		    savepath = savepath, stopping_epochs = stopping_epochs,
		    use_gpu = use_gpu, verbose = verbose
		  )
  else
      estimator = train(
        estimator, theta_train, theta_val, Z_train, Z_val,
        M,
        # loss = loss, optimiser = optimiser, #TODO uncomment when I figure it out
        epochs = epochs, batchsize = batchsize,
		    savepath = savepath, stopping_epochs = stopping_epochs,
		    use_gpu = use_gpu, verbose = verbose
		  )
  end

  estimator
  ',
   estimator = estimator, theta_train = theta_train,
   theta_val = theta_val, Z_train = Z_train, Z_val = Z_val,
   M = M,
   # loss = loss, optimiser = optimiser, #TODO uncomment when I figure it out
   epochs = epochs, batchsize = batchsize,
   savepath = savepath, stopping_epochs = stopping_epochs,
   use_gpu = use_gpu, verbose = verbose
  )

  return(estimator)
}


#' @title load the weights of the best neural estimator
#' @export
loadbestweights <- function(estimator, path) {
  # path: absolute path to the runs folder
  juliaLet(
    '
    using NeuralEstimators
    using Flux
    Flux.loadparams!(estimator, NeuralEstimators.loadbestweights(path))
    estimator
    ',
    estimator = estimator, path = path
    )
}


#' @title create a piecewise neural estimator
#' @export
PiecewiseEstimator <- function(estimators, mchange) {
  juliaLet('PiecewiseEstimator(estimators, mchange)', estimators = estimators, mchange = mchange)
}


# TODO a few other arguments that we need to add here (estimator/parameter names, save, etc.)
#' @title assess one or more (neural) estimators
#' @export
assess <- function(
  estimators, # should be a list of estimators
  parameters,
  Z,
  # xi = NULL,
  # use_xi = FALSE,
  use_gpu = TRUE,
  verbose = TRUE
) {

  assessment <- juliaLet('

  using NeuralEstimators
  using Flux

  # Convert parameters and data to Float32 for computational efficiency
  Z = broadcast.(z -> Float32.(z), Z)
  parameters = broadcast.(Float32, parameters)

  assessment = assess(
        estimators, parameters, Z,
		    use_gpu = use_gpu, verbose = verbose
		  )
  ', estimators = estimators, parameters = parameters, Z = Z, use_gpu = TRUE, verbose = TRUE)

  estimates <- juliaLet('assessment.θandθ̂', assessment = assessment)
  runtimes  <- juliaLet('assessment.runtime', assessment = assessment)

  estimates <- as.data.frame(estimates)
  runtimes  <- as.data.frame(runtimes)

  list(estimates = estimates, runtimes = runtimes)
}



#' @title apply a neural estimator to the data Z
#' @export
estimate <- function(estimator, Z) {
  thetahat <- juliaLet('

  using NeuralEstimators
  using Flux

  # Convert data to Float32 for computational efficiency
  Z = broadcast.(Float32, Z)

  # Move the data to the GPU # TODO need to check if the gpu is available and if estimator is on the gpu
  # Z = Z |> gpu

  theta_hat = estimator(Z)

  # move to the cpu
  theta_hat = theta_hat |> cpu
  ', estimator = estimator, Z = Z)
  return(thetahat)
}



