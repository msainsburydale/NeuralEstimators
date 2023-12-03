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
#' @param loss the loss function. It can be a string 'absolute-error' or 'squared-error', in which case the loss function will be the mean-absolute-error or mean-squared-error loss. Otherwise, one may provide a custom loss function as a string of Julia code, which will be converted to a Julia function using \code{juliaEval()}
#' @param learning_rate the learning rate for the optimiser ADAM (default 1e-4)
#' @param epochs the number of epochs
#' @param stopping_epochs cease training if the risk doesn't improve in this number of epochs (default 5)
#' @param batchsize the batchsize to use when performing stochastic gradient descent
#' @param savepath path to save the trained estimator and other information; if null (default), nothing is saved
#' @param use_gpu a boolean indicating whether to use the GPU if it is available (default true)
#' @param verbose a boolean indicating whether information should be printed to the console during training
#' @param epochs_per_Z_refresh integer indicating how often to refresh the training data
#' @param epochs_per_theta_refresh integer indicating how often to refresh the training parameters; must be a multiple of \code{epochs_per_Z_refresh}
#' @param simulate_just_in_time  flag indicating whether we should simulate "just-in-time", in the sense that only a \code{batchsize} number of parameter vectors and corresponding data are in memory at a given time
#' @return a trained neural estimator or, if \code{m} is provided a vector, a list of trained neural estimators
#' @export
#' @seealso [assess()] for assessing an estimator post training, and [estimate()] for applying an estimator to observed data
#' @examples
#' \dontrun{
#' # Construct a neural Bayes estimator for replicated univariate Gaussian 
#' # data with unknown mean and standard deviation. 
#' 
#' # Load R and Julia packages
#' library("NeuralEstimators")
#' library("JuliaConnectoR")
#' juliaEval("using NeuralEstimators; using Flux; using Distributions")
#' 
#' # Define the neural-network architecture
#' estimator <- juliaEval('
#'  d = 1    # dimension of each replicate
#'  p = 2    # number of parameters in the model
#'  w = 32   # width of each layer
#'  psi = Chain(Dense(d, w, relu), Dense(w, w, relu))
#'  phi = Chain(Dense(w, w, relu), Dense(w, p))
#'  estimator = DeepSet(psi, phi)
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
#' # Train the estimator using simulation on-the-fly
#' m <- 30 # number of iid replicates
#' estimator <- train(estimator, sampler = sampler, simulator = simulator, m = m)
#' 
#' # Train the estimator using fixed parameter and data sets 
#' theta_train <- sampler(10000)
#' theta_val   <- sampler(2000)
#' Z_train <- simulator(theta_train, m)
#' Z_val   <- simulator(theta_val, m)
#' estimator <- train(estimator, 
#'                    theta_train = theta_train, 
#'                    theta_val = theta_val, 
#'                    Z_train = Z_train, 
#'                    Z_val = Z_val)
#' 
#' ##### Simulation on-the-fly using Julia functions ####
#' 
#' # Defining the sampler and simulator in Julia can improve computational 
#' # efficiency by avoiding the overhead of communicating between R and Julia. 
#' # Julia is also fast (comparable to C) and so it can be useful to define 
#' # these functions in Julia when they involve for loops. 
#' 
#' # Parameter sampler
#' sampler <- juliaEval("
#'       function sampler(K)
#'       	μ = rand(Normal(0, 1), K)
#'       	σ = rand(Gamma(1), K)
#'       	θ = hcat(μ, σ)'
#'       	return θ
#'       end")
#' 
#' # Data simulator
#' simulator <- juliaEval("
#'       function simulator(θ_matrix, m)
#'       	Z = [rand(Normal(θ[1], θ[2]), 1, m) for θ ∈ eachcol(θ_matrix)]
#'       	return Z
#'       end")
#' 
#' # Train the estimator
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
                  learning_rate = 1e-4,
                  epochs = 100,
                  batchsize = 32,
                  savepath = "",
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

  # Metaprogramming: load Julia packages and add keyword arguments that are applicable to all methods of train()
  code <- paste(
  "
  using NeuralEstimators
  using Flux
  
  estimator = ",
     train_code,
    "
    loss = loss,
    optimiser = ADAM(learning_rate),
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

# TODO should clean these functions up... bit untidy with how it is
# TODO need to add testing of these functions

#' @title load the weights of a neural estimator
#' @param estimator the neural estimator that we wish to load weights into
#' @param filename file (including absolute path) of the neural-network weights saved as a \code{bson} file
#' @export
loadweights <- function(estimator, filename) {
  juliaLet(
    '
    using NeuralEstimators
    using Flux
    Flux.loadparams!(estimator, NeuralEstimators.loadweights(filename))
    estimator
    ',
    estimator = estimator, filename = filename
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
#' @seealso [assess()]
#' @export
risk <- function(df, 
                 loss = function(x, y) abs(x - y), 
                 average_over_parameters = TRUE, 
                 average_over_sample_sizes = TRUE) {
  
  # TODO add checks that df contains the correct columns
  
  truth <- NULL # Setting the variables to NULL first to appease CRAN checks (see https://stackoverflow.com/questions/9439256/how-can-i-handle-r-cmd-check-no-visible-binding-for-global-variable-notes-when)

  # Determine which variables we are grouping by
  grouping_variables = "estimator"
  if (!average_over_parameters) grouping_variables <- c(grouping_variables, "parameter")
  if (!average_over_sample_sizes) grouping_variables <- c(grouping_variables, "m")
  
  # Compute the risk and its standard deviation
  df %>%
    mutate(loss = loss(estimate, truth)) %>%
    group_by(across(all_of(grouping_variables))) %>%
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
#' @seealso [risk()] for computing the empirical Bayes risk from the returned object, and [plotdistribution()] and [plotrisk()] for functions that visualise the results contained in the \code{estimates} data frame described above
#' @export
assess <- function(
  estimators, # should be a list of estimators
  parameters,
  Z,
  # NB not sure if there is a reason to implement xi
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
#' @param estimator a neural estimator
#' @param Z data to apply the estimator to; it's format should be amenable to the architecture of \code{estimator}
#' @param use_gpu a boolean indicating whether to use the GPU if it is available (default true)
#' @return a matrix of parameter estimates (i.e., \code{estimator} applied to \code{Z})
#' @export
#' @examples
#' \dontrun{
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
#' estimate(estimator, Z)}
estimate <- function(estimator, Z, use_gpu = TRUE) {

  if (!is.list(Z)) Z <- list(Z)

  thetahat <- juliaLet('
  using NeuralEstimators
  using Flux

  # Estimate the parameters
  # TODO should change to estimate_in_batches() at some point
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

#' @title bootstrap
#' 
#' @description Generate \code{B} bootstrap estimates from a neural estimator
#' 
#' Non-parametric bootstrap is facilitated by setting the data \code{Z} to 
#' be a single data set.  Parametric bootstrap is facilitated by setting the 
#' data \code{Z} to be multiple simulated data sets stored as a list whose 
#' length implicitly defines \code{B}. 
#'
#' @param estimator a neural estimator
#' @param Z either simulated data of length B or a single observed data set, which will be bootstrap sampled B times to generate B bootstrap estimates
#' @param B number of bootstrap estimates (default 400)
#' @param blocks integer vector specifying the blocks in non-parameteric bootstrapping (default \code{NULL}). For example, with 5 replicates, the first two corresponding to block 1 and the remaining three corresponding to block 2, blocks should be \code{c(1,1,2,2,2)}. The bootstrap sampling algorithm aims to produce bootstrap data sets that are of a similar size to \code{Z}, but this can only be achieved exactly if all blocks are equal in length.
#' @param use_gpu a boolean indicating whether to use the GPU if it is available (default \code{TRUE})
#' @return p × B matrix, where p is the number of parameters in the model and B is the number of bootstrap samples
#' @export
#' @examples
#' \dontrun{
#' library("NeuralEstimators")
#' library("JuliaConnectoR")
#' 
#' ## Observed data: m independent replicates of a N(0, 1) random variable
#' m = 100
#' Z = t(rnorm(m))
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
#'   phi = Chain(Dense(w, w, relu), Dense(w, p, exp))
#'   estimator = DeepSet(psi, phi)
#' ')
#'
#' ## Non-parametric bootstrap
#' bootstrap(estimator, Z = Z)
#' bootstrap(estimator, Z = Z, blocks = rep(1:5, each = m/5))
#'
#' ## Parametric bootstrap 
#' thetahat = estimate(estimator, Z)  # estimated parameters
#' B = 400
#' Z = lapply(1:B, function(b) t(rnorm(m, mean = thetahat[1], sd = thetahat[2])))
#' bootstrap(estimator, Z = Z)}
bootstrap <- function(estimator,
                      Z,
                      B = 400,
                      blocks = NULL,
                      use_gpu = TRUE
                      ) {

  B <- as.integer(B)
  if (!is.list(Z)) Z <- list(Z)

  if (length(Z) > 1) {
    #NB Can alternatively just use estimateinbatches() since thats all we're doing here
    thetahat <- juliaLet('
      using NeuralEstimators
      Z = broadcast.(Float32, Z)
      bootstrap(estimator, parameters, Z, use_gpu = use_gpu)',
                         estimator=estimator, parameters=matrix(1), Z=Z, use_gpu=use_gpu # NB dummy value of parameters provided here, since we don't actually need it. 
    )
  } else {
    thetahat <- juliaLet('
      using NeuralEstimators
      Z = broadcast.(Float32, Z)
      bootstrap(estimator, Z, use_gpu = use_gpu, B = B, blocks = blocks)',
      estimator=estimator, Z=Z, use_gpu=use_gpu, blocks=blocks, B=B
    )
  }

  return(thetahat)
}
