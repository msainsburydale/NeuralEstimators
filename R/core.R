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

#' @title CPU device
#'
#' @description A CPU compute device from [MLDataDevices.jl](https://github.com/LuxDL/MLDataDevices.jl) for the `device` argument of [train()].
#'
#' @param ... arguments passed to the Julia function.
#' @return a Julia device object
#' @export
#' @seealso [gpu_device()], [reactant_device()], [train()]
cpu_device <- function(...) {
  .getNeuralEstimators()
  juliaCall("NeuralEstimators.cpu_device", ...)
}

#' @title GPU device
#'
#' @description A GPU compute device from [MLDataDevices.jl](https://github.com/LuxDL/MLDataDevices.jl) for the `device` argument of [train()].
#'
#' @param ... arguments passed to the Julia function (e.g. `force`).
#' @return a Julia device object
#' @export
#' @seealso [cpu_device()], [reactant_device()], [train()]
gpu_device <- function(...) {
  .getNeuralEstimators()
  juliaCall("NeuralEstimators.gpu_device", ...)
}

#' @title Reactant/XLA device
#'
#' @description A Reactant/XLA compute device from [MLDataDevices.jl](https://github.com/LuxDL/MLDataDevices.jl) for the `device` argument of [train()]. Requires Lux.
#'
#' @param ... arguments passed to the Julia function (e.g. `force`).
#' @return a Julia device object
#' @export
#' @seealso [cpu_device()], [gpu_device()], [train()]
reactant_device <- function(...) {
  .getNeuralEstimators()
  juliaCall("NeuralEstimators.reactant_device", ...)
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
#' @param sampler_args a list of positional arguments passed to the parameter sampler; if provided, `sampler` is called as \code{sampler(K, sampler_args...)}.
#' @param sampler_kwargs a list of positional arguments passed to the parameter sampler; if provided, `sampler` is called as \code{sampler(K; sampler_kwargs...)}.
#' @param simulator_args a list of positional arguments passed to the data simulator; if provided, `simulator` is called as \code{simulator(theta, simulator_args...)}.
#' @param simulator_kwargs a list of positional arguments passed to the data simulator; if provided, `simulator` is called as \code{simulator(theta; simulator_kwargs...)}.
#' @param m deprecated; use \code{simulator_args}
#' @param K the number of parameter vectors sampled in the training set at each epoch
#' @param K_val the number of parameter vectors in the validation set; if \code{NULL} (default), the Julia default is used
#' @param loss the loss function: a string ('absolute-error' for mean-absolute-error loss or 'squared-error' for mean-squared-error loss), or a string of Julia code defining the loss function. For some classes of estimators (e.g., `PosteriorEstimator`, `QuantileEstimator`, `RatioEstimator`), the loss function does not need to be specified.
#' @param learning_rate the initial learning rate for the optimiser ADAM (default 5e-4) 
#' @param epochs the number of epochs to train the neural network. An epoch is one complete pass through the entire training data set when doing stochastic gradient descent.
#' @param stopping_epochs cease training if the risk doesn't improve in this number of epochs (default 5).
#' @param batchsize the batchsize to use when performing stochastic gradient descent, that is, the number of training samples processed between each update of the neural-network parameters. 
#' @param savepath path to save the trained estimator and other information; if null (default), nothing is saved. Otherwise, the neural-network parameters (i.e., the weights and biases) will be saved during training as `bson` files; the risk function evaluated over the training and validation sets will also be saved, in the first and second columns of `loss_per_epoch.csv`, respectively; the best parameters (as measured by validation risk) will be saved as `best_network.bson`. 
#' @param device the compute device, as a Julia object from [MLDataDevices.jl](https://github.com/LuxDL/MLDataDevices.jl), e.g. [cpu_device()], [gpu_device()], or [reactant_device()] (the latter requires Lux). If \code{NULL} (default), the device is inferred from \code{use_gpu}. Takes priority over \code{use_gpu}.
#' @param use_gpu a boolean indicating whether to use the GPU if one is available (ignored if \code{device} is provided)
#' @param shuffle whether to shuffle the training set at each epoch
#' @param partial whether to include the final incomplete batch; if \code{NULL} (default), the Julia default is used
#' @param freeze_summary_network if \code{TRUE} and the estimator has a summary network, freeze those parameters during training
#' @param verbose a boolean indicating whether information, including empirical risk values and timings, should be printed to the console during training.
#' @param epochs_per_refresh integer indicating how often to refresh the training data
#' @param epochs_per_Z_refresh deprecated; use \code{epochs_per_refresh}
#' @param epochs_per_theta_refresh integer indicating how often to refresh the training parameters; must be a multiple of \code{epochs_per_refresh}. If \code{NULL} (default), it is set equal to \code{epochs_per_refresh}
#' @param simulate_just_in_time  flag indicating whether we should simulate "just-in-time", in the sense that only a \code{batchsize} number of parameter vectors and corresponding data are in memory at a given time
#' @param ... additional keyword arguments passed to the Julia version of [`train()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/training)
#' @return a trained neural estimator or, if \code{m} is a vector, a list of trained neural estimators
#' @export
#' @seealso the Julia version of [`train()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/training), [assess()] for assessing an estimator post training, and [infer()]/[estimate()]/[sampleposterior()] for making inference with observed data
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
#' estimator <- train(estimator, sampler = sampler, simulator = simulator, m = m)
#' }
train <- function(estimator,
                  sampler = NULL,   
                  simulator = NULL, 
                  theta_train = NULL,
                  theta_val = NULL,
                  Z_train = NULL,
                  Z_val = NULL,
                  K = 10000,
                  K_val = NULL,
                  m = NULL, # deprecated
                  sampler_args = NULL,
                  sampler_kwargs = NULL, 
                  simulator_args = NULL,
                  simulator_kwargs = NULL,
                  loss = "absolute-error",
                  learning_rate = 5e-4,
                  epochs = 100,
                  batchsize = 32,
                  savepath = NULL,
                  stopping_epochs = 5,
                  epochs_per_refresh = 1,
                  epochs_per_Z_refresh = NULL, # deprecated
                  epochs_per_theta_refresh = NULL,  
                  simulate_just_in_time = FALSE,
                  device = NULL,
                  use_gpu = TRUE,
                  shuffle = TRUE,
                  partial = NULL,
                  freeze_summary_network = FALSE,
                  verbose = TRUE,
                  ...
                  ) {

  # Deprecation coercion 
  if (!is.null(m)) {
    # warning("The argument `m` in `train()` is deprecated; please use `simulator_kwargs`") #TODO how should R users pass this though? Should be a NamedTuple
    simulator_args <- juliaLet('(m,)', m = as.integer(m))
  }
  if (!is.null(epochs_per_Z_refresh)) {
    warning("`epochs_per_Z_refresh` is deprecated; use `epochs_per_refresh`")
    epochs_per_refresh <- epochs_per_Z_refresh
  }
  
  # Convert numbers that should be integers (so that the user can write 32 rather than 32L)
  K <- as.integer(K)
  epochs <- as.integer(epochs)
  batchsize <- as.integer(batchsize)
  stopping_epochs <- as.integer(stopping_epochs)
  epochs_per_refresh <- as.integer(epochs_per_refresh)
  if (!is.null(K_val)) K_val <- as.integer(K_val)
  if (!is.null(epochs_per_theta_refresh)) epochs_per_theta_refresh <- as.integer(epochs_per_theta_refresh)
  
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
  # this using RCall (see https://juliainterop.github.io/RCall.jl/stable/).
  # Since JuliaConnectoR creates a separate R environment when Julia is 
  # initialised, we must use the macro @rput to move the R functions to this 
  # separate R environment before the R functions can be invoked. 
  if (!is.null(sampler) && !("JLFUN" %in% names(attributes(sampler)))) {
    tryCatch( { juliaEval("using RCall") }, error = function(e) "using R functions to perform 'on-the-fly' simulation requires the user to have installed the Julia package RCall")
    juliaLet('using RCall; @rput sampler', sampler = sampler)
    sampler <- juliaEval('
        using RCall
        sampler(K) = rcopy(R"sampler($K)")
        sampler(K, sampler_args) = rcopy(R"sampler($K, $sampler_args)")
                         ')
  }
  if (!is.null(simulator) && !("JLFUN" %in% names(attributes(simulator)))) {
    tryCatch( { juliaEval("using RCall") }, error = function(e) "using R functions to perform 'on-the-fly' simulation requires the user to have installed the Julia package RCall")
    juliaLet('using RCall; @rput simulator', simulator = simulator)
    simulator <- juliaEval('using RCall; simulator(theta, m) = rcopy(R"simulator($theta, $m)")')
  }
  
  # Metaprogramming: All other keyword arguments for on-the-fly simulation 
  if (!is.null(simulator)) {
    if (!is.null(simulator_args))   train_code <- paste(train_code, "simulator_args = simulator_args,")
    if (!is.null(simulator_kwargs)) train_code <- paste(train_code, "simulator_kwargs = simulator_kwargs,")
    train_code <- paste(train_code, "epochs_per_refresh = epochs_per_refresh, simulate_just_in_time = simulate_just_in_time,")
  }
  if (!is.null(sampler)) {
    if (!is.null(sampler_args))   train_code <- paste(train_code, "sampler_args = sampler_args,")
    if (!is.null(sampler_kwargs)) train_code <- paste(train_code, "sampler_kwargs = sampler_kwargs,")
    train_code <- paste(train_code, "K = K,")
    if (!is.null(K_val)) train_code <- paste(train_code, "K_val = K_val,")
    if (!is.null(epochs_per_theta_refresh)) train_code <- paste(train_code, "epochs_per_theta_refresh = epochs_per_theta_refresh,")
  }
  
  # Identify which loss function we are using; if it is a string that matches
  # absolute-error or squared-error, convert it to the Julia function
  # corresponding to those loss functions. Otherwise, pass it in unchanged,
  # so that the user can provide a Julia function defining a custom loss function.
  if (loss == "absolute-error") {
    loss = juliaEval('mae(yhat, y) = mean(abs.(yhat .- y))')
  } else if (loss == "squared-error") {
    loss = juliaEval('mse(yhat, y) = mean(abs2.(yhat .- y))')
  } else {
    loss = juliaEval(loss)
  }

  dots <- list(...)
  if (length(dots) && (is.null(names(dots)) || any(!nzchar(names(dots))))) {
    stop("Additional arguments to train() must be named")
  }
  
  # Metaprogramming: load Julia packages and add keyword arguments that are applicable to all methods of train()
  extra_kwargs <- ""
  if (!is.null(partial)) extra_kwargs <- paste(extra_kwargs, "partial = partial,")
  if (length(dots)) extra_kwargs <- paste(extra_kwargs, paste(sprintf("%s = %s,", names(dots), names(dots)), collapse = " "))

  code <- paste(
  "
  using NeuralEstimators
  using Optimisers: Adam
  using Statistics: mean
  
  estimator = ",
     train_code,
    "
    loss = loss,
    optimiser = Optimisers.Adam(learning_rate), 
    epochs = epochs,
    batchsize = batchsize,
    savepath = savepath,
    stopping_epochs = stopping_epochs,
    device = device,
    use_gpu = use_gpu,
    shuffle = shuffle,
    freeze_summary_network = freeze_summary_network,
    verbose = verbose,",
    extra_kwargs,
    "
  )

  estimator")

  # Run the Julia code and pass the arguments from R to Julia
  jl_args <- list(
     code,
     estimator = estimator, 
     sampler = sampler, simulator = simulator,
     theta_train = theta_train, theta_val = theta_val, 
     Z_train = Z_train, Z_val = Z_val, 
     K = K, 
     sampler_args = sampler_args,
     sampler_kwargs = sampler_kwargs,
     simulator_args = simulator_args,
     simulator_kwargs = simulator_kwargs,
     loss = loss,
     learning_rate = learning_rate,
     epochs = epochs,
     batchsize = batchsize,
     savepath = savepath,
     stopping_epochs = stopping_epochs,
     device = device,
     use_gpu = use_gpu,
     shuffle = shuffle,
     freeze_summary_network = freeze_summary_network,
     verbose = verbose, 
     epochs_per_refresh = epochs_per_refresh,      
     simulate_just_in_time = simulate_just_in_time
  )
  if (!is.null(K_val)) jl_args$K_val <- K_val
  if (!is.null(epochs_per_theta_refresh)) jl_args$epochs_per_theta_refresh <- epochs_per_theta_refresh
  if (!is.null(partial)) jl_args$partial <- partial
  if (length(dots)) {
    overlap <- intersect(names(dots), names(jl_args))
    if (length(overlap)) stop("Argument(s) already specified: ", paste(overlap, collapse = ", "))
    jl_args <- c(jl_args, dots)
  }
  estimator <- do.call(juliaLet, jl_args)

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
  juliaEval('using NeuralEstimators')
  juliaEval('using BSON: @load')
  is_lux <- juliaLet(
    '
    estimator isa LuxEstimator || begin
      lux = get(Base.loaded_modules, NeuralEstimators._LUX_UUID, nothing)
      !isnothing(lux) && NeuralEstimators._is_lux_network(estimator, lux)
    end
    ',
    estimator = estimator
  )
  if (isTRUE(is_lux)) {
    juliaLet(
      '
      @load filename ps st
      inner = estimator isa LuxEstimator ? estimator.estimator : estimator
      LuxEstimator(inner, ps, st)
      ',
      estimator = estimator, filename = filename
    )
  } else {
    juliaEval('using Flux')
    juliaLet(
      '
      @load filename model_state
      Flux.loadmodel!(estimator, model_state)
      estimator
      ',
      estimator = estimator, filename = filename
    )
  }
}

#' @title save the state of a neural estimator
#' @description Save the state of a neural estimator (e.g., optimised neural-network parameters). Supports both Flux and Lux estimators.
#' @param estimator the neural estimator that we wish to save
#' @param filename file in which to save the neural-network state as a \code{bson} file
#' @return No return value, called for side effects
#' @export
savestate <- function(estimator, filename) {
  juliaEval('using NeuralEstimators')
  juliaEval('using BSON: @save')
  is_lux <- juliaLet('estimator isa LuxEstimator', estimator = estimator)
  if (isTRUE(is_lux)) {
    juliaLet(
      '
      ps = estimator.ps
      st = estimator.st
      @save filename ps st
      ',
      estimator = estimator, filename = filename
    )
  } else {
    juliaEval('using Flux')
    juliaLet(
      '
      model_state = Flux.state(estimator)
      @save filename model_state
      ',
      estimator = estimator, filename = filename
    )
  }
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
  
  if (is.list(assessment)) df <- if (!is.null(assessment$df)) assessment$df else assessment$estimates
  if (is.data.frame(assessment)) df <- assessment
  
  truth <- NULL # Setting the variables to NULL first to appease CRAN checks (see https://stackoverflow.com/questions/9439256/how-can-i-handle-r-cmd-check-no-visible-binding-for-global-variable-notes-when)
  
  # Determine which variables we are grouping by
  grouping_variables <- intersect(
    c(
      "estimator",
      if (!average_over_parameters) "parameter",
      if (!average_over_sample_sizes) "m"
    ),
    names(df)
  )
  
  
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
#' @param estimator a neural estimator (or a list of neural estimators)
#' @param parameters true parameters, stored as a \eqn{d\times K}{dxK} matrix, where \eqn{d} is the dimension of the parameter vector and \eqn{K} is the number of sampled parameter vectors
#' @param Z data simulated conditionally on the \code{parameters}. If \code{length(Z)} > K, the parameter matrix will be recycled by horizontal concatenation as `parameters = parameters[, rep(1:K, J)]`, where `J = length(Z) / K`
#' @param ... additional keyword arguments passed to the Julia version of [`assess()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/assessment)
#' @return a list with the following elements:
#' \itemize{
#' \item{\code{estimates}: a long-form data frame with columns
#'   \itemize{
#'   \item{"parameter"; the name of the parameter}
#'   \item{"truth"; the true value of the parameter}
#'   \item{"estimate"; the estimated value of the parameter}
#'   \item{"k"; the index of the parameter vector in the test set}
#'   \item{"j"; the index of the data set}
#'   }
#'   For an \code{IntervalEstimator}, \code{estimate} is replaced by columns \code{lower} and \code{upper}. For a \code{QuantileEstimator}, there is an additional column \code{prob} giving the probability level of each quantile estimate.}
#' \item{\code{runtimes}: a data frame giving the total time taken for each estimator}
#' \item{\code{samples} (posterior and ratio estimators only): posterior samples stored as a long-form Julia \code{DataFrame} with columns \code{parameter}, \code{truth}, \code{k}, and \code{j} as above, plus \code{draw} (the index of the draw) and \code{value} (the sampled value). This object is not converted to an R data frame, which is slow for large sample collections; extract or summarise it in Julia if needed.}
#' }
#' @seealso [risk()], [rmse()], [bias()], [plotestimates()], and [plotdistribution()] for computing various empirical diagnostics and visualisations from an object returned by `assess()`
#' @export
assess <- function(
    estimator, 
    parameters,
    Z,
    ...
) {
  
  if (is.vector(parameters)) parameters <- t(parameters)
  
  NE <- .getNeuralEstimators()
  assessment <- NE$assess(estimator, parameters, Z, ...)
  
  estimates <- juliaLet('assessment.estimates', assessment = assessment)
  runtimes  <- juliaLet('assessment.runtime', assessment = assessment)
  
  estimates <- as.data.frame(estimates)
  runtimes  <- as.data.frame(runtimes)
  output <- list(estimates = estimates, runtimes = runtimes)
  
  # Add assessment.samples if it is available
  if (juliaLet("hasproperty(assessment, :samples) && !isnothing(assessment.samples)", assessment = assessment)) {
    samples <- juliaLet('assessment.samples', assessment = assessment)
    if (!is.null(samples)) {
      #samples <- as.data.frame(samples) # NB this conversion takes a really long time, so just provide the samples directly as a Julia object
      output <- c(output, list(samples = samples))
    }
  }
  
  return(output)
}

#' @title sampleposterior
#' 
#' @description Samples from the approximate posterior distribution given data `Z`. 
#'
#' @param estimator a neural posterior or likelihood-to-evidence-ratio estimator
#' @param Z data in a format amenable to the neural-network architecture of `estimator`
#' @param N number of approximate posterior samples to draw
#' @param ... additional keyword arguments passed to the Julia version of [`sampleposterior()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/inference#Inference-with-observed-data)
#' @return a \eqn{d \times N \times K}{d x N x K} array of posterior samples, where \eqn{d} is the dimension of the parameter vector, \code{N} is the number of draws, and \eqn{K} is the number of independent data sets in \code{Z} (so a single data set yields a \eqn{d \times N \times 1}{d x N x 1} array)
#' @seealso [infer()] for a unified inference interface, and [estimate()] for making inference with neural Bayes estimators
#' @export
sampleposterior <- function(estimator, Z, N = 1000, ...) {
  NE <- .getNeuralEstimators()
  samples <- NE$sampleposterior(estimator, Z, N = as.integer(N), ...)
  juliaLet('Float64.(samples)', samples = samples)
}

#' @title estimate
#'
#' @description Apply a neural Bayes estimator to data
#'
#' @param estimator a neural estimator that can be applied to data in a call of the form `estimator(Z)`
#' @param Z data in a format amenable to the neural-network architecture of `estimator`
#' @param batchsize the batch size for applying `estimator` to `Z`
#' @param ... additional keyword arguments passed to the Julia version of [`estimate()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/inference#Inference-with-observed-data)
#' @return a matrix of outputs resulting from applying `estimator` to `Z`
#' @seealso [infer()] for a unified inference interface, and [sampleposterior()] for making inference with neural posterior or likelihood-to-evidence-ratio estimators
#' @export
estimate <- function(estimator, Z, batchsize = 32, ...) {
  NE <- .getNeuralEstimators()
  thetahat <- NE$estimate(estimator, Z, batchsize = as.integer(batchsize), ...)
  thetahat <- juliaLet('Float64.(thetahat)', thetahat = thetahat) # convert to regular matrix and Float64
  return(thetahat)
}

#' @title infer
#'
#' @description Unified inference interface that dispatches to [estimate()] for neural Bayes estimators (e.g., `PointEstimator`) and [sampleposterior()] for posterior or ratio estimators.
#'
#' @param estimator a neural estimator
#' @param Z data in a format amenable to the neural-network architecture of `estimator`
#' @param ... additional keyword arguments passed to the Julia version of [`infer()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/inference#Inference-with-observed-data) (and hence to [estimate()] or [sampleposterior()])
#' @return For neural Bayes estimators, a matrix of point estimates (see [estimate()]). For posterior or ratio estimators, a \eqn{d \times N \times K}{d x N x K} array of posterior samples (see [sampleposterior()]).
#' @seealso [estimate()], [sampleposterior()]
#' @export
infer <- function(estimator, Z, ...) {
  NE <- .getNeuralEstimators()
  dots <- list(...)
  if (!is.null(dots$batchsize)) dots$batchsize <- as.integer(dots$batchsize)
  if (!is.null(dots$N)) dots$N <- as.integer(dots$N)

  result <- do.call(NE$infer, c(list(estimator, Z), dots))
  juliaLet('Float64.(result)', result = result)
}

#' @title logratio
#'
#' @description Apply a neural ratio estimator to data
#'
#' @param estimator a neural ratio estimator
#' @param Z data in a format amenable to the neural-network architecture of `estimator`
#' @param grid matrix of parameter values, where each column is a parameter configuration
#' @param batchsize the batch size
#' @param ... additional keyword arguments passed to the Julia version of [`logratio()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/inference#Inference-with-observed-data)
#' @return A matrix of log ratios with one row per data set and one column per grid point
#' @seealso [sampleposterior()] and [infer()] for making posterior inferences
#' @export
logratio <- function(estimator, Z, grid, batchsize = 32, ...) {
  NE <- .getNeuralEstimators()
  log_ratios <- NE$logratio(estimator, Z, grid = grid, batchsize = as.integer(batchsize), ...)
  log_ratios <- juliaLet('Float64.(log_ratios)', log_ratios = log_ratios) # convert to regular matrix and Float64
  return(log_ratios)
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
