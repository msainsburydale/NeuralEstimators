# Same as JuliaConnectoR::juliaSetupOk(), but also check minimum version number
.juliaSetupOk <- function(version = "1") {
  
  juliaCmd <- NULL
  try({
    # juliaCmd <- JuliaConnectoR:::getJuliaExecutablePath()
    juliaCmd <- asNamespace("JuliaConnectoR")$getJuliaExecutablePath()
  })
  if (is.null(juliaCmd)) {
    message("Julia not found")
    return(FALSE)
  }
  
  # juliaVersion <- JuliaConnectoR:::getJuliaVersionViaCmd(juliaCmd)
  juliaVersion <-  asNamespace("JuliaConnectoR")$getJuliaVersionViaCmd(juliaCmd)
  if (is.null(juliaVersion)) {
    message("Julia could not be started")
    return(FALSE)
  }
  
  # Convert versions to numeric vectors
  juliaVersionNum <- as.integer(unlist(strsplit(juliaVersion, ".", fixed = TRUE)))
  requiredVersionNum <- as.integer(unlist(strsplit(version, ".", fixed = TRUE)))
  
  # Pad versions with zeros for safe comparison (e.g., "1.11" -> c(1,11,0) if necessary)
  lengthDiff <- length(requiredVersionNum) - length(juliaVersionNum)
  if (lengthDiff > 0) {
    juliaVersionNum <- c(juliaVersionNum, rep(0, lengthDiff))
  } else if (lengthDiff < 0) {
    requiredVersionNum <- c(requiredVersionNum, rep(0, -lengthDiff))
  }
  
  # Compare versions element-wise
  if (isTRUE(all(juliaVersionNum >= requiredVersionNum))) {
    return(TRUE)
  } else {
    message("Julia version must be at least ", version)
    return(FALSE)
  }
}

.getNeuralEstimators <- function() {
  if (.juliaSetupOk() && juliaEval('"NeuralEstimators" in keys(Pkg.project().dependencies)')) {
    juliaImport("NeuralEstimators") 
  } else {
    stop("Julia package NeuralEstimators.jl not found. If it is installed but not detected, please report this issue to the package maintainer.")
  }
}

#' @title Train a neural estimator
#' 
#' @description The function caters for different variants of "on-the-fly" simulation. 
#' Specifically, a \code{sampler} can be provided to continuously sample new 
#' parameter vectors from the prior, and a \code{simulator} can be provided to 
#' continuously simulate new data conditional on the parameters. If provided 
#' with specific sets of parameters (\code{theta_train} and \code{theta_val}) 
#' and/or data (\code{Z_train} and \code{Z_val}), they will be held fixed during 
#' training.
#' 
#' Note that using \code{R} functions to perform "on-the-fly" simulation requires the user to have installed the Julia package \code{RCall}.
#'
#' @param estimator a neural estimator
#' @param sampler a function that takes an integer \code{K}, samples \code{K} parameter vectors from the prior, and returns them as a px\code{K} matrix
#' @param simulator a function that takes a px\code{K} matrix of parameters and an integer \code{m}, and returns \code{K} simulated data sets each containing \code{m} independent replicates
#' @param theta_train a set of parameters used for updating the estimator using stochastic gradient descent
#' @param theta_val a set of parameters used for monitoring the performance of the estimator during training
#' @param Z_train a simulated data set used for updating the estimator using stochastic gradient descent
#' @param Z_val a simulated data set used for monitoring the performance of the estimator during training
#' @param m vector of sample sizes. If \code{NULL} (default), a single neural estimator is trained, with the sample size inferred from \code{Z_val}. If \code{m} is a vector of integers, a sequence of neural estimators is constructed for each sample size; see the Julia documentation for \code{trainx()} for further details
#' @param M deprecated; use \code{m}
#' @param K the number of parameter vectors sampled in the training set at each epoch; the size of the validation set is set to \code{K}/5.
#' @param xi a list of objects used for data simulation (e.g., distance matrices); if it is provided, the parameter sampler is called as \code{sampler(K, xi)}.
#' @param loss the loss function: a string ('absolute-error' for mean-absolute-error loss or 'squared-error' for mean-squared-error loss), or a string of Julia code defining the loss function. For some classes of estimators (e.g., `PosteriorEstimator`, `QuantileEstimator`, `RatioEstimator`), the loss function does not need to be specified.
#' @param learning_rate the initial learning rate for the optimiser ADAM (default 5e-4) 
#' @param epochs the number of epochs to train the neural network. An epoch is one complete pass through the entire training data set when doing stochastic gradient descent.
#' @param stopping_epochs cease training if the risk doesn't improve in this number of epochs (default 5).
#' @param batchsize the batchsize to use when performing stochastic gradient descent, that is, the number of training samples processed between each update of the neural-network parameters. 
#' @param savepath path to save the trained estimator and other information; if null (default), nothing is saved. Otherwise, the neural-network parameters (i.e., the weights and biases) will be saved during training as `bson` files; the risk function evaluated over the training and validation sets will also be saved, in the first and second columns of `loss_per_epoch.csv`, respectively; the best parameters (as measured by validation risk) will be saved as `best_network.bson`. 
#' @param use_gpu a boolean indicating whether to use the GPU if one is available 
#' @param verbose a boolean indicating whether information, including empirical risk values and timings, should be printed to the console during training.
#' @param epochs_per_Z_refresh integer indicating how often to refresh the training data
#' @param epochs_per_theta_refresh integer indicating how often to refresh the training parameters; must be a multiple of \code{epochs_per_Z_refresh}
#' @param simulate_just_in_time  flag indicating whether we should simulate "just-in-time", in the sense that only a \code{batchsize} number of parameter vectors and corresponding data are in memory at a given time
#' @return a trained neural estimator or, if \code{m} is a vector, a list of trained neural estimators
#' @export
#' @seealso [assess()] for assessing an estimator post training, and [estimate()]/[sampleposterior()] for making inference with observed data
#' @examples
#' \dontrun{
#' # Construct a neural Bayes estimator for replicated univariate Gaussian 
#' # data with unknown mean and standard deviation. 
#' 
#' # Load R and Julia packages
#' library("NeuralEstimators")
#' library("JuliaConnectoR")
#' juliaEval("using NeuralEstimators, Flux")
#' 
#' # Define the neural-network architecture
#' estimator <- juliaEval('
#'  n = 1    # dimension of each replicate
#'  d = 2    # number of parameters in the model
#'  w = 32   # width of each hidden layer
#'  psi = Chain(Dense(n, w, relu), Dense(w, w, relu))
#'  phi = Chain(Dense(w, w, relu), Dense(w, d))
#'  deepset = DeepSet(psi, phi)
#'  estimator = PointEstimator(deepset)
#' ')
#' 
#' # Sampler from the prior
#' sampler <- function(K) {
#'   mu    <- rnorm(K)      # Gaussian prior for the mean
#'   sigma <- rgamma(K, 1)  # Gamma prior for the standard deviation
#'   theta <- matrix(c(mu, sigma), byrow = TRUE, ncol = K)
#'   return(theta)
#' }
#' 
#' # Data simulator
#' simulator <- function(theta_set, m) {
#'   apply(theta_set, 2, function(theta) {
#'     t(rnorm(m, theta[1], theta[2]))
#'   }, simplify = FALSE)
#' }
#' 
#' # Train using fixed parameter and data sets 
#' theta_train <- sampler(10000)
#' theta_val   <- sampler(2000)
#' m <- 30 # number of iid replicates
#' Z_train <- simulator(theta_train, m)
#' Z_val   <- simulator(theta_val, m)
#' estimator <- train(estimator, 
#'                    theta_train = theta_train, 
#'                    theta_val = theta_val, 
#'                    Z_train = Z_train, 
#'                    Z_val = Z_val)
#'                    
#' ##### Simulation on-the-fly using R functions ####
#' 
#' juliaEval("using RCall") # requires the Julia package RCall
#' estimator <- train(estimator, sampler = sampler, simulator = simulator, m = m)
#' 
#' ##### Simulation on-the-fly using Julia functions ####
#' 
#' # Defining the sampler and simulator in Julia can improve computational 
#' # efficiency by avoiding the overhead of communicating between R and Julia. 
#' 
#' juliaEval("using Distributions")
#' 
#' # Parameter sampler
#' sampler <- juliaEval("
#'       function sampler(K)
#'       	mu = rand(Normal(0, 1), K)
#'       	sigma = rand(Gamma(1), K)
#'       	theta = hcat(mu, sigma)'
#'       	return theta
#'       end")
#' 
#' # Data simulator
#' simulator <- juliaEval("
#'       function simulator(theta_matrix, m)
#'       	Z = [rand(Normal(theta[1], theta[2]), 1, m) for theta in eachcol(theta_matrix)]
#'       	return Z
#'       end")
#' 
#' # Train
#' estimator <- train(estimator, sampler = sampler, simulator = simulator, m = m)}
train <- function(estimator,
                  sampler = NULL,   
                  simulator = NULL, 
                  theta_train = NULL,
                  theta_val = NULL,
                  Z_train = NULL,
                  Z_val = NULL,
                  m = NULL, M = NULL, # M is a deprecated argument
                  K = 10000,        
                  xi = NULL,        
                  loss = "absolute-error",
                  learning_rate = 5e-4,
                  epochs = 100,
                  batchsize = 32,
                  savepath = NULL,
                  stopping_epochs = 5,
                  epochs_per_Z_refresh = 1,      
                  epochs_per_theta_refresh = 1,  
                  simulate_just_in_time = FALSE, 
                  use_gpu = TRUE,
                  verbose = TRUE
                  ) {

  # Deprecation coercion 
  if (!is.null(M)) {
    warning("The argument `M` in `train()` is deprecated; please use `m`")
    m <- M 
  }
  
  # Convert numbers that should be integers (so that the user can write 32 rather than 32L)
  K <- as.integer(K)
  epochs <- as.integer(epochs)
  batchsize <- as.integer(batchsize)
  stopping_epochs <- as.integer(stopping_epochs)
  epochs_per_Z_refresh <- as.integer(epochs_per_Z_refresh)
  epochs_per_theta_refresh <- as.integer(epochs_per_theta_refresh)
  
  # Coerce theta_train and theta_val to 1xK matrices if given as K-vectors (which will often be the case in single-parameter settings)
  if (is.vector(theta_train)) theta_train <- t(theta_train)
  if (is.vector(theta_val)) theta_val <- t(theta_val)
  
  # logic check
  if (!is.null(sampler) && (!is.null(Z_train) || !is.null(Z_val))) stop("One cannot combine continuous resampling of the parameters through `sampler` with fixed simulated data sets, `Z_train` and `Z_val`")
  
  # Metaprogramming: Define the Julia code based on the given arguments
  train_code <- "train(estimator,"
  if (is.null(sampler)) {
    if (is.null(theta_train) || is.null(theta_val)) stop("A parameter `sampler` or sampled parameter sets `theta_train` and `theta_val` must be provided")
    train_code <- paste(train_code, "theta_train, theta_val,")
  } else {
    if (!is.null(theta_train) || !is.null(theta_val)) stop("Only one of `sampler` or `theta_train` and `theta_val` should be provided")
    train_code <- paste(train_code, "sampler,")
  }
  if (is.null(simulator)) {
    if (is.null(Z_train) || is.null(Z_val)) stop("A data `simulator` or simulated data sets `Z_train` and `Z_val` must be provided")
    train_code <- paste(train_code, "Z_train, Z_val,")
  } else {
    if (!is.null(Z_train) || !is.null(Z_val)) stop("Only one of `simulator` or `Z_train` and `Z_val` should be provided")
    train_code <- paste(train_code, "simulator,")
  }
  
  # If `sampler` and `simulator` are not Julia functions (i.e., they lack "JLFUN" 
  # attributes), then we need to define Julia functions that invoke them. We do 
  # this using the package RCall (see https://juliainterop.github.io/RCall.jl/stable/).
  # Further, since JuliaConnectoR creates a separate R environment when Julia is 
  # initialised, we must use the macro @rput to move the R functions to this 
  # separate R environment before the R functions can be invoked. 
  if (!is.null(sampler) && !("JLFUN" %in% names(attributes(sampler)))) {
    tryCatch( { juliaEval("using RCall") }, error = function(e) "using R functions to perform 'on-the-fly' simulation requires the user to have installed the Julia package RCall")
    juliaLet('using RCall; @rput sampler', sampler = sampler)
    sampler <- juliaEval('
        using RCall
        sampler(K) = rcopy(R"sampler($K)")
        sampler(K, xi) = rcopy(R"sampler($K, $xi)")
                       ')
  }
  if (!is.null(simulator) && !("JLFUN" %in% names(attributes(simulator)))) {
    tryCatch( { juliaEval("using RCall") }, error = function(e) "using R functions to perform 'on-the-fly' simulation requires the user to have installed the Julia package RCall")
    juliaLet('using RCall; @rput simulator', simulator = simulator)
    simulator <- juliaEval('using RCall; simulator(theta, m) = rcopy(R"simulator($theta, $m)")')
  }

  # Metaprogramming: Define the Julia code based on the value of m
  if (is.null(m)) {
    if (!is.null(simulator)) stop("Since a data `simulator` was provided, the number of independent replicates `m` to simulate must also be provided")  
  } else {
    m <- as.integer(m)
    if (length(m) == 1) {
      train_code <- paste(train_code, "m = m,")
    } else {
      train_code <- sub("train", "trainx", train_code)
      train_code <- paste(train_code, "m,")
    }
  } 
  
  # Metaprogramming: All other keyword arguments for on-the-fly simulation 
  if (!is.null(simulator)) train_code <- paste(train_code, "epochs_per_Z_refresh = epochs_per_Z_refresh, simulate_just_in_time = simulate_just_in_time,")
  if (!is.null(sampler)) train_code <- paste(train_code, "K = K, xi = xi, epochs_per_theta_refresh = epochs_per_theta_refresh,")

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
  
  # Omit the loss function for certain classes of neural estimators
  omit_loss <- juliaLet('typeof(estimator) <: Union{PosteriorEstimator, RatioEstimator, IntervalEstimator, QuantileEstimator, QuantileEstimatorDiscrete, QuantileEstimatorContinuous}', estimator = estimator)
  loss_code <- if (omit_loss) "" else "loss = loss,"
  
  # Metaprogramming: load Julia packages and add keyword arguments that are applicable to all methods of train()
  code <- paste(
  "
  using NeuralEstimators, Flux
  
  estimator = ",
     train_code, 
     loss_code,
    "
    optimiser = Flux.setup(Flux.Adam(learning_rate), estimator), 
    epochs = epochs,
    batchsize = batchsize,
    savepath = savepath,
    stopping_epochs = stopping_epochs,
    use_gpu = use_gpu,
    verbose = verbose
  )

  estimator")

  # Run the Julia code and pass the arguments from R to Julia
  estimator = juliaLet(
     code,
     estimator = estimator, 
     sampler = sampler, simulator = simulator,
     theta_train = theta_train, theta_val = theta_val, 
     Z_train = Z_train, Z_val = Z_val, 
     m = m,
     K = K, 
     xi = xi,
     loss = loss,
     learning_rate = learning_rate,
     epochs = epochs,
     batchsize = batchsize,
     savepath = savepath,
     stopping_epochs = stopping_epochs,
     use_gpu = use_gpu,
     verbose = verbose, 
     epochs_per_theta_refresh = epochs_per_theta_refresh,  
     epochs_per_Z_refresh = epochs_per_Z_refresh,      
     simulate_just_in_time = simulate_just_in_time
  )

  return(estimator)
}

#' @title Load a saved state of a neural estimator
#' 
#' @description Load a saved state of a neural estimator (e.g., optimised neural-network parameters). Useful for amortised inference, whereby a neural network is trained once and then used repeatedly to make inference with new data sets.
#'
#' @param estimator the neural estimator that we wish to load the state into
#' @param filename file name (including path) of the neural-network state stored in a \code{bson} file
#' @return `estimator` updated with the saved state 
#' @export
loadstate <- function(estimator, filename) {
  juliaEval('using NeuralEstimators, Flux')
  juliaEval('using BSON: @load')
  juliaLet(
    '
    @load filename model_state
    Flux.loadmodel!(estimator, model_state)
    estimator
    ',
    estimator = estimator, filename = filename
  )
}

#' @title save the state of a neural estimator
#' @param estimator the neural estimator that we wish to save
#' @param filename file in which to save the neural-network state as a \code{bson} file
#' @return No return value, called for side effects
#' @export
savestate <- function(estimator, filename) {
  juliaEval('using NeuralEstimators, Flux')
  juliaEval('using BSON: @save')
  juliaLet(
    '
    model_state = Flux.state(estimator)
    @save filename model_state
    ',
    estimator = estimator, filename = filename
  )
}

#' @title computes a Monte Carlo approximation of an estimator's Bayes risk
#' @param assessment an object returned by [assess()] 
#' @param loss a binary operator defining the loss function (default absolute-error loss)
#' @param average_over_parameters if \code{TRUE}, the loss is averaged over all parameters; otherwise (default), the loss is averaged over each parameter separately
#' @param average_over_sample_sizes if \code{TRUE} (default), the loss is averaged over all sample sizes (the column \code{m} in \code{df}); otherwise, the loss is averaged over each sample size separately
#' @return a dataframe giving an estimate of the Bayes risk
#' @seealso [assess()], [bias()], [rmse()]
#' @export
risk <- function(assessment, 
                 loss = function(x, y) abs(x - y), 
                 average_over_parameters = FALSE, 
                 average_over_sample_sizes = TRUE
                 ) {
  
  if (is.list(assessment)) df <- assessment$estimates
  if (is.data.frame(assessment)) df <- assessment
  
  truth <- NULL # Setting the variables to NULL first to appease CRAN checks (see https://stackoverflow.com/questions/9439256/how-can-i-handle-r-cmd-check-no-visible-binding-for-global-variable-notes-when)
  
  # Determine which variables we are grouping by
  grouping_variables = "estimator"
  if (!average_over_parameters) grouping_variables <- c(grouping_variables, "parameter")
  if (!average_over_sample_sizes) grouping_variables <- c(grouping_variables, "m")
  
  # Compute the risk 
  dplyr::mutate(df, loss = loss(estimate, truth)) %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(grouping_variables))) %>%
    dplyr::summarise(risk = mean(loss))
}

#' @title computes a Monte Carlo approximation of an estimator's bias
#' @inheritParams risk 
#' @param ... optional arguments inherited from `risk()` (excluding the argument `loss`)
#' @return a dataframe giving the estimated bias
#' @seealso [assess()], [risk()], [rmse()]
#' @export
bias <- function(assessment, ...) {
  df <- risk(assessment, loss = function(x, y) x - y, ...)
  names(df)[names(df) == "risk"] <- "bias"
  return(df)
}

#' @title computes a Monte Carlo approximation of an estimator's root-mean-square error (RMSE)
#' @inheritParams risk 
#' @param ... optional arguments inherited from `risk()` (excluding the argument `loss`)
#' @return a dataframe giving the estimated RMSE
#' @seealso [assess()], [bias()], [risk()]
#' @export
rmse <- function(assessment, ...) {
  df <- risk(assessment, loss = function(x, y) (x - y)^2, ...)
  df$risk <- sqrt(df$risk)
  names(df)[names(df) == "risk"] <- "rmse"
  return(df)
}

#' @title assess a neural estimator
#' @param estimators a neural estimator (or a list of neural estimators)
#' @param parameters true parameters, stored as a \eqn{d\times K}{dxK} matrix, where \eqn{d} is the dimension of the parameter vector and \eqn{K} is the number of sampled parameter vectors
#' @param Z data simulated conditionally on the \code{parameters}. If \code{length(Z)} > K, the parameter matrix will be recycled by horizontal concatenation as `parameters = parameters[, rep(1:K, J)]`, where `J = length(Z) / K`
#' @param estimator_names list of names of the estimators (sensible defaults provided)
#' @param parameter_names list of names of the parameters (sensible defaults provided)
#' @param use_gpu a boolean indicating whether to use the GPU if it is available (default true)
#' @param verbose a boolean indicating whether information should be printed to the console
#' @return a list of two data frames: \code{runtimes} contains the
#' total time taken for each estimator, while \code{df} is a long-form
#' data frame with columns:
#' \itemize{
#' \item{"estimator"; the name of the estimator}
#' \item{"parameter"; the name of the parameter}
#' \item{"truth"; the true value of the parameter}
#' \item{"estimate"; the estimated value of the parameter}
#' \item{"m"; the sample size (number of iid replicates)}
#' \item{"k"; the index of the parameter vector in the test set}
#' \item{"j"; the index of the data set}
#' }
#' @seealso [risk()], [rmse()], [bias()], [plotestimates()], and [plotdistribution()] for computing various empirical diagnostics and visualisations from an object returned by `assess()`
#' @export
assess <- function(
  estimators, 
  parameters,
  Z,
  estimator_names = NULL,
  parameter_names = NULL,
  use_gpu = TRUE,
  verbose = TRUE
) {

  if (!is.list(estimators)) estimators <- list(estimators)
  if (is.vector(parameters)) parameters <- t(parameters)

  # Metaprogramming: Define the Julia code based on the value of the arguments
  estimator_names_code <- if (!is.null(estimator_names)) " estimator_names = estimator_names, " else ""
  parameter_names_code <- if (!is.null(parameter_names)) " parameter_names = parameter_names, " else ""

  if (length(estimator_names) == 1 & !is.list(estimator_names)) estimator_names <- list(estimator_names)
  if (length(parameter_names) == 1 & !is.list(parameter_names)) parameter_names <- list(parameter_names)

  code <- paste(
  "
  using NeuralEstimators, Flux

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
#' @description Apply a neural estimator to data
#'
#' @param estimator a neural estimator that can be applied to data in a call of the form `estimator(Z)`
#' @param Z data in a format amenable to the neural-network architecture of `estimator`
#' @param X additional inputs to the neural network; if provided, the call will be of the form `estimator((Z, X))`
#' @param batchsize the batch size for applying `estimator` to `Z`. Batching occurs only if `Z` is a list, indicating multiple data sets 
#' @param use_gpu boolean indicating whether to use the GPU if it is available
#' @return a matrix of outputs resulting from applying `estimator` to `Z` (and possibly `X`)
#' @seealso [sampleposterior()] for making inference with neural posterior or likelihood-to-evidence-ratio estimators
#' @export
estimate <- function(estimator, Z, X = NULL, batchsize = 32, use_gpu = TRUE) {
  NE <- .getNeuralEstimators()
  thetahat <- NE$estimate(estimator, Z, X, use_gpu = use_gpu, batchsize = as.integer(batchsize))
  thetahat <- juliaLet('using Flux: f64; f64(thetahat)', thetahat = thetahat) # convert to regular matrix and Float64
  return(thetahat)
}

#' @title bootstrap
#' 
#' @description Compute bootstrap estimates from a neural point estimator
#'
#' @param estimator a neural point estimator
#' @param Z either a list of data sets simulated conditionally on the fitted parameters (parametric bootstrap); or a single observed data set containing independent replicates, which will be sampled with replacement `B` times (non-parametric bootstrap)
#' @param B number of non-parametric bootstrap samples
#' @param blocks integer vector specifying the blocks in non-parameteric bootstrap. For example, with 5 replicates, the first two corresponding to block 1 and the remaining three corresponding to block 2, `blocks` should be \code{c(1,1,2,2,2)}
#' @param use_gpu boolean indicating whether to use the GPU if it is available
#' @return d × `B` matrix, where d is the dimension of the parameter vector 
#' @export
bootstrap <- function(estimator,
                      Z,
                      B = 400,
                      blocks = NULL,
                      use_gpu = TRUE
                      ) {

  B <- as.integer(B)
  if (!is.list(Z)) Z <- list(Z)

  if (length(Z) > 1) {
    #NB Just using estimate() since that is all that needs to be done here
    thetahat <- estimate(estimator, Z, use_gpu = use_gpu)
  } else {
    NE <- .getNeuralEstimators()
    thetahat <- NE$bootstrap(estimator, Z, use_gpu = use_gpu, B = B, blocks = blocks)
  }

  return(thetahat)
}

#' @title sampleposterior
#' 
#' @description Samples from the approximate posterior distribution given data `Z`. 
#'
#' @param estimator a neural posterior or likelihood-to-evidence-ratio estimator
#' @param Z data in a format amenable to the neural-network architecture of `estimator`
#' @param N number of approximate posterior samples to draw
#' @param ... additional keyword arguments passed to the Julia version of [`sampleposterior()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/core/#NeuralEstimators.sampleposterior), applicable when `estimator` is a likelihood-to-evidence-ratio estimator
#' @return a d × `N` matrix of posterior samples, where d is the dimension of the parameter vector. If `Z` is a list containing multiple data sets, a list of matrices will be returned
#' @seealso [estimate()] for making inference with neural Bayes estimators
#' @export
sampleposterior <- function(estimator, Z, N = 1000, ...) {
  N <- as.integer(N)
  NE <- .getNeuralEstimators()
  NE$sampleposterior(estimator, Z, N, ...)
}

#' @title posteriormode
#' 
#' @description Computes the (approximate) posterior mode (maximum a posteriori estimate) given data `Z`.  
#' 
#' @inheritParams sampleposterior 
#' @param ... additional keyword arguments passed to the Julia version of [`posteriormode()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/core/#NeuralEstimators.posteriormode)
#' @return a d × K matrix of posterior samples, where d is the dimension of the parameter vector and K is the number of data sets provided in `Z`
#' @seealso [sampleposterior()] for sampling from the approximate posterior distribution
#' @export
posteriormode <- function(estimator, Z, ...) {
  NE <- .getNeuralEstimators()
  NE$posteriormode(estimator, Z, ...)
}
