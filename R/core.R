# Wrapper functions around the core functions in NeuralEstimators.jl.

#TODO on-the-fly simulation

#' @title Train a neural estimator
#'
#' @description Train a neural estimator with architecture given by \code{estimator}.
#'
#' Note that "on-the-fly" simulation is currently not implemented using the R interface.
#'
#' @param estimator the neural estimator
#' @param theta_train the set of parameters used for updating the estimator using stochastic gradient descent
#' @param theta_val the set of parameters used for monitoring the performance of the estimator during training
#' @param Z_train the data used for updating the estimator using stochastic gradient descent
#' @param Z_val the data used for monitoring the performance of the estimator during training
#' @param M vector of sample sizes. If null (default), a single neural estimator is trained, with the sample size inferred from \code{Z_val}. If \code{M} is a vector of integers, a sequence of neural estimators is constructed for each sample size; see the Julia documentation for \code{trainx()} for details
#' @param loss the loss function. It can be a string 'absolute-error' or 'squared-error', in which case the loss function will be the mean absolute-error or squared-error loss. Otherwise, one may provide a custom loss function as Julia code, which will be converted to a Julia function using \code{juliaEval()}
#' @param learning the learning rate for the optimiser ADAM (default 1e-4)
#' @param epochs the number of epochs
#' @param stopping_epochs cease training if the risk doesn't improve in this number of epochs (default 5)
#' @param batchsize the batchsize to use when performing stochastic gradient descent; reducing this can alleviate memory pressure
#' @param savepath path to save the trained estimator and other information; if null (default), nothing is saved
#' @param use_gpu a boolean indicating whether to use the GPU if it is available (default true)
#' @param verbose a boolean indicating whether information should be printed to the console during training
#' @return a trained neural estimator or, if \code{M} is provided, a list of trained neural estimators
#' @export
#' @seealso [assess()] for assessing an estimator post training, and [estimate()] for applying an estimator to observed data
#' @examples
#' # Construct a neural Bayes estimator for univariate Gaussian data
#' # with unknown mean and standard deviation, based on m = 15 iid replicates.
#'
#' library("NeuralEstimators")
#' library("JuliaConnectoR")
#'
#' # Sample from the prior
#' prior <- function(K) {
#'   mu    <- rnorm(K)
#'   sigma <- rgamma(K, 1)
#'   theta <- matrix(c(mu, sigma), byrow = TRUE, ncol = K)
#'   return(theta)
#' }
#' theta_train = prior(10000)
#' theta_val   = prior(2000)
#'
#' # Simulate univariate data conditional on the above parameter vectors
#' simulate <- function(theta_set, m) {
#'  apply(theta_set, 2, function(theta) {
#'    t(rnorm(m, theta[1], theta[2]))
#'  }, simplify = FALSE)
#' }
#' m <- 30
#' Z_train <- simulate(theta_train, m)
#' Z_val   <- simulate(theta_val, m)
#'
#' # Define the neural-network architecture
#' estimator <- juliaEval('
#'   using NeuralEstimators
#'   using Flux
#'   p = 2    # number of parameters in the model
#'   w = 32   # width of each layer
#'   psi = Chain(Dense(1, w, relu), Dense(w, w, relu))
#'   phi = Chain(Dense(w, w, relu), Dense(w, p))
#'   estimator = DeepSet(psi, phi)
#' ')
#'
#' # Train a neural estimator
#' estimator <- train(
#'   estimator,
#'   theta_train = theta_train,
#'   theta_val   = theta_val,
#'   Z_train = Z_train,
#'   Z_val   = Z_val,
#'   epochs = 50
#'   )
#'   
#'  # List of neural estimators, one trained with 15 replicates, and another with 30 replicates
#' estimators <- train(
#'   estimator,
#'   theta_train = theta_train,
#'   theta_val   = theta_val,
#'   Z_train = Z_train,
#'   Z_val   = Z_val,
#'   M = c(15, 30), 
#'   epochs = 2
#' )
train <- function(estimator,
                  theta_train,
                  theta_val,
                  Z_train,
                  Z_val,
                  M = NULL, # if M is left NULL, it is ignored
                  loss = "absolute-error",
                  learning_rate = 1e-4,
                  epochs = 100,
                  batchsize = 32,
                  savepath = "",
                  stopping_epochs = 5,
                  use_gpu = TRUE,
                  verbose = TRUE
                  ) {

  # Convert numbers that should be integers (so that the user can write 32 rather than 32L)
  epochs <- as.integer(epochs)
  batchsize <- as.integer(batchsize)
  stopping_epochs <- as.integer(stopping_epochs)

  # Metaprogramming: Define the Julia code based on the value of M
  if (is.null(M)) {
    train_code <- "train(estimator, theta_train, theta_val, Z_train, Z_val,"
  } else {
    M <- sapply(M, as.integer)
    train_code <- "trainx(estimator, theta_train, theta_val, Z_train, Z_val, M,"
  }

  # Identify which loss function we are using; if it is a string that matches
  # absolute-error or squared-error, convert it to the Julia function
  # corresponding to those loss functions. Otherwise, pass it in unchanged,
  # so that the user can provide a Julia function defining a custom loss function.
  if (loss == "absolute-error") {
    loss = juliaEval('Flux.Losses.mae')
  } else if (loss == "squared-error") {
    loss = juliaEval('Flux.Losses.mse')
  } else {
    loss = juliaEval(loss)
  }

  code <- paste0(
  "
  using NeuralEstimators
  using Flux

  # Convert parameters and data to Float32 for computational efficiency
  theta_train = broadcast.(Float32, theta_train)
  theta_val = broadcast.(Float32, theta_val)
  Z_train = broadcast.(Float32, Z_train)
  Z_val   = broadcast.(Float32, Z_val)

  estimator = ",
     train_code,
    "
    loss = loss,
    #optimiser = ADAM(learning_rate),
    epochs = epochs,
    batchsize = batchsize,
    savepath = savepath,
    stopping_epochs = stopping_epochs,
    use_gpu = use_gpu,
    verbose = verbose
  )

  estimator ")


  estimator = juliaLet(
     code,
     estimator = estimator, theta_train = theta_train,
     theta_val = theta_val, Z_train = Z_train, Z_val = Z_val, M = M,
     loss = loss,
     learning_rate = learning_rate,
     epochs = epochs,
     batchsize = batchsize,
     savepath = savepath,
     stopping_epochs = stopping_epochs,
     use_gpu = use_gpu,
     verbose = verbose
  )

  return(estimator)
}


#' @title load the weights of a neural estimator
#' @param estimator the neural estimator that we wish to load weights into
#' @param file file (including absolute path) of the neural-network weights saved as a \code{bson} file
#' @export
loadweights <- function(estimator, filename) {
  juliaLet(
    '
    using NeuralEstimators
    using Flux
    Flux.loadparams!(estimator, NeuralEstimators.loadweights(file))
    estimator
    ',
    estimator = estimator, path = path
  )
}

#' @title load the weights of the best neural estimator
#' @param estimator the neural point estimator that we wish to load weights into
#' @param path absolute path to the folder containing the saved neural-network weights, saved as \code{best_network.bson}
#' @export
loadbestweights <- function(estimator, path) {
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

#' @title computes a Monte Carlo approximation of the Bayes risk
#' @param df the \code{estimates} dataframe returned by a call to \code{assess()}
#' @param loss a binary operator defining the loss function (default absolute-error loss)
#' @param average_over_parameters if \code{TRUE} (default), the loss is averaged over all parameters; otherwise, the loss is averaged over each parameter separately
#' @param average_over_sample_sizes if \code{TRUE} (default), the loss is averaged over all sample sizes (the column \code{m} in \code{df}); otherwise, the loss is averaged over each sample size separately
#' @return a dataframe giving the estimated risk and an estimate of its standard deviation
#' @export
risk <- function(df, 
                 loss = function(x, y) abs(x - y), 
                 average_over_parameters = TRUE, 
                 average_over_sample_sizes = TRUE) {

  # Determine which variables we are grouping by
  grouping_variables = "estimator"
  if (!average_over_parameters) grouping_variables <- c(grouping_variables, "parameter")
  if (!average_over_sample_sizes) grouping_variables <- c(grouping_variables, "m")
  
  # Compute the risk and its standard deviation
  df %>%
    mutate(loss = loss(estimate, truth)) %>%
    group_by(across(grouping_variables)) %>%
    summarise(risk = mean(loss), risk_sd = sd(loss)/sqrt(length(loss))) 
}





#' @title assess a neural estimator
#' @param estimators a list of (neural) estimators
#' @param parameters true parameters, stored as a pxK matrix, where p is the number of parameters in the statistical model and K is the number of sampled parameter vectors
#' @param Z data simulated conditionally on the \code{parameters}. If \code{Z} contains more data sets than parameter vectors, the parameter matrix will be recycled by horizontal concatenation.
#' @param estimator_names list of names of the estimators (sensible defaults provided)
#' @param parameter_names list of names of the parameters (sensible defaults provided)
#' @param use_gpu a boolean indicating whether to use the GPU if it is available (default true)
#' @param verbose a boolean indicating whether information should be printed to the console
#' @return a list of two data frames: \code{runtimes}, contains the
#' total time taken for each estimator, while \code{estimates} is a long-form
#' data frame with columns:
#' \itemize{
#' \item{"estimator"}{ the name of the estimator}
#' \item{"parameter"}{ the name of the parameter}
#' \item{"truth"}{ the true value of the parameter}
#' \item{"estimate"}{ the estimated value of the parameter}
#' \item{"m"}{ the sample size (number of iid replicates)}
#' \item{"k"}{ the index of the parameter vector in the test set}
#' \item{"j"}{ the index of the data set}
#' }
#' @seealso [plotdistribution()] and [plotrisk()] for functions that visualise the results contained in the \code{estimates} data frame described above
#' @export
assess <- function(
  estimators, # should be a list of estimators
  parameters,
  Z,
  # NB don't bother with xi, might be too complicated for now
  # xi = NULL,
  # use_xi = FALSE,
  estimator_names = NULL,
  parameter_names = NULL,
  use_gpu = TRUE,
  verbose = TRUE
) {

  if (!is.list(estimators)) estimators <- list(estimators)

  # Metaprogramming: Define the Julia code based on the value of the arguments
  estimator_names_code <- if (!is.null(estimator_names)) " estimator_names = estimator_names, " else ""
  parameter_names_code <- if (!is.null(parameter_names)) " parameter_names = parameter_names, " else ""

  if (length(estimator_names) == 1 & !is.list(estimator_names)) estimator_names <- list(estimator_names)
  if (length(parameter_names) == 1 & !is.list(parameter_names)) parameter_names <- list(parameter_names)

  code <- paste0(
  "
  using NeuralEstimators
  using Flux

  # Convert parameters and data to Float32 for computational efficiency
  Z = broadcast.(z -> Float32.(z), Z)
  parameters = broadcast.(Float32, parameters)

  assessment = assess(
        estimators, parameters, Z,",
		    estimator_names_code, parameter_names_code,
		    "use_gpu = use_gpu, verbose = verbose
		  )
  ")


  assessment <- juliaLet(code, estimators = estimators, parameters = parameters, Z = Z,
                         use_gpu = TRUE, verbose = TRUE,
                         estimator_names = estimator_names,
                         parameter_names = parameter_names)

  estimates <- juliaLet('assessment.df', assessment = assessment)
  runtimes  <- juliaLet('assessment.runtime', assessment = assessment)

  estimates <- as.data.frame(estimates)
  runtimes  <- as.data.frame(runtimes)

  list(estimates = estimates, runtimes = runtimes)
}


#' @title estimate
#'
#' @description estimate parameters from observed data using a neural estimator
#'
#'
#' @param estimator a neural estimator
#' @param Z data to apply the estimator to; it's format should be amenable to the architecture of \code{estimator}
#' @param use_gpu a boolean indicating whether to use the GPU if it is available (default true)
#' @return a matrix of parameter estimates (i.e., \code{estimator} applied to \code{Z})
#' @export
#' @examples
#' library("NeuralEstimators")
#' library("JuliaConnectoR")
#'
#' ## Observed data: 100 replicates of a univariate random variable
#' Z = matrix(rnorm(100), nrow = 1)
#'
#' ## Construct the estimator
#' estimator <- juliaEval('
#'   using NeuralEstimators
#'   using Flux
#'
#'   p = 2    # number of parameters in the statistical model
#'   w = 32   # number of neurons in each layer
#'
#'   psi = Chain(Dense(1, w, relu), Dense(w, w, relu))
#'   phi = Chain(Dense(w, w, relu), Dense(w, p))
#'   estimator = DeepSet(psi, phi)
#' ')
#'
#' ## Apply the estimator
#' estimate(estimator, Z)
estimate <- function(estimator, Z, use_gpu = TRUE) {

  if (!is.list(Z)) Z <- list(Z)

  thetahat <- juliaLet('
  using NeuralEstimators
  using Flux

  # Estimate the parameters
  theta_hat = NeuralEstimators._runondevice(estimator, Z, use_gpu)
  # if use_gpu
  #   Z = Z |> gpu
  #   estimator = estimator |> gpu
  # end
  # theta_hat = estimator(Z)

  # Move back to the cpu and convert to a regular matrix
  theta_hat = theta_hat |> cpu
  theta_hat = Float64.(theta_hat)
  theta_hat
  ', estimator = estimator, Z = Z, use_gpu=use_gpu)
  return(thetahat)
}

#TODO
#' @title bootstrap
#' @description Generate bootstrap estimates from an estimator
#'
#' Parametric bootstrapping is facilitated by passing multiple simulated data set, Z, which should be stored as a list and whose length implicitly defines B.
#'
#' Non-parametric bootstrapping is facilitated by passing a single data set, Z. The argument \code{blocks} caters for block bootstrapping, and it should be a vector of integers specifying the block for each replicate. For example, with 5 replicates, the first two corresponding to block 1 and the remaining three corresponding to block 2, blocks should be \code{c(1, 1, 2, 2, 2)}. The resampling algorithm aims to produce resampled data sets that are of a similar size to Z, but this can only be achieved exactly if all blocks are equal in length.
#'
#' @param estimator a neural estimator
#' @param Z either simulated data of length B or a single observed data set, which will be bootstrap sampled B times to generate B bootstrap estimates
#' @param parameters a single parameter configuration (default \code{NULL})
#' @param B number of bootstrap samples (default 400)
#' @param blocks integer vector specifying the blocks in non-parameteric bootstrapping (default \code{NULL}). For example, with 5 replicates, the first two corresponding to block 1 and the remaining three corresponding to block 2, blocks should be \code{c(1, 1, 2, 2, 2)}
#' @param use_gpu a boolean indicating whether to use the GPU if it is available (default true)
#' @return p × B matrix, where p is the number of parameters in the model and B is the number of bootstrap samples
#' @export
#' @examples
#' library("NeuralEstimators")
#' library("JuliaConnectoR")
#'
#' ## Observed data: 100 replicates of a univariate random variable
#' Z = matrix(rnorm(100), nrow = 1)
#'
#' ## Construct the estimator
#' estimator <- juliaEval('
#'   using NeuralEstimators
#'   using Flux
#'
#'   p = 2    # number of parameters in the statistical model
#'   w = 32   # number of neurons in each layer
#'
#'   psi = Chain(Dense(1, w, relu), Dense(w, w, relu))
#'   phi = Chain(Dense(w, w, relu), Dense(w, p))
#'   estimator = DeepSet(psi, phi)
#' ')
#'
#' ## Non-parametric bootstrap
#' bootstrap(estimator, Z = Z)
#' bootstrap(estimator, Z = Z, blocks = rep(1:5, each = 20))
#'
#' ## Parametric bootstrap (pretend that the following data generating process
#' involves simulation from the model given estimated parameters)
#' B = 400
#' Z = lapply(1:B, function(b) matrix(rnorm(100), nrow = 1))
#' bootstrap(estimator, Z = Z)
bootstrap <- function(estimator,
                      Z,
                      B = 400,
                      blocks = NULL,
                      use_gpu = TRUE
                      ) {

  B <- as.integer(B)

  if (length(Z) > 1) {
    thetahat <- juliaLet('
      using NeuralEstimators
      bootstrap(estimator, parameters, Z, use_gpu = use_gpu)',
      estimator=estimator, parameters=matrix(1), Z=Z, use_gpu=use_gpu # NB dummy value of parameters provided here, since we don't actually need it. Probably should change the Julia code to not require parameters.
      )
  } else {
    thetahat <- juliaLet('
      using NeuralEstimators
      bootstrap(estimator, Z, use_gpu = use_gpu, B = B, blocks = blocks)',
      estimator=estimator, Z=Z, use_gpu=use_gpu, blocks=blocks, B=B
    )
  }

  return(thetahat)
}



