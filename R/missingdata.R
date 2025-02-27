#' @title encodedata
#'
#' @description For data `Z` with missing (`NA`) entries, computes an augmented data set (U, W) where W encodes the missingness pattern as an indicator vector and U is the original data Z with missing entries replaced by a fixed constant `c`.
#'
#' @param Z data containing `NA` entries
#' @param c fixed constant with which to replace `NA` entries
#' @return Augmented data set (U, W). If `Z` is provided as a list, the return type will be a `JuliaProxy` object; these objects can be indexed in the usual manner using `[[`, or converted to an R object using `juliaGet()` (note however that `juliaGet()` can be slow for large data sets). 
#' @seealso the Julia version of [`encodedata()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/utility/#NeuralEstimators.encodedata)
#' @export
#' @examples
#' \dontrun{
#' library("NeuralEstimators")
#' Z <- matrix(c(1, 2, NA, NA, 5, 6, 7, NA, 9), nrow = 3)
#' encodedata(Z)
#' encodedata(list(Z, Z))}
encodedata <- function(Z, c = 0) {
  juliaEval('using NeuralEstimators')
  if (is.list(Z)) {
    UW <- juliaLet('encodedata.(Z; c = c)', Z = Z, c = c)
  } else {
    UW <- juliaLet('encodedata(Z; c = c)', Z = Z, c = c)
  }
  return(UW)
}

#' @title tanhloss
#' @description For \code{k} > 0, defines Julia code that defines the loss function,
#' \deqn{L(\hat{\theta}, \theta) = \tanh\left(\frac{|\hat{\theta} - \theta|}{k}\right),}
#' which approximates the 0-1 loss as \code{k} tends to zero. 
#' 
#' The resulting string is intended to be used in the function \code{\link{train}}, but can also be converted to a callable function using \code{juliaEval}. 
#' @param k Positive numeric value that controls the smoothness of the approximation.
#' @return String defining the tanh loss function in Julia code.
#' @export 
tanhloss <- function(k) paste0("(x, y) -> tanhloss(x, y, ", k, ")")

#' @title spatialgraph
#'
#' @description Constructs a graph object for use in a graph neural network (GNN). 
#'
#' @param S Spatial locations, provided as:
#'   - An \eqn{n\times 2}{n x 2} matrix when locations are fixed across replicates, where \eqn{n} is the number of spatial locations. 
#'   - A list of \eqn{n_i \times 2}{ni x 2} matrices when locations vary across replicates.
#'   - A list of the above elements (i.e., a list of matrices or a list of lists of matrices) when constructing graphs from multiple data sets.
#' @param Z Spatial data, provided as:
#'   - An \eqn{n\times m}{n x m} matrix when locations are fixed, where \eqn{m} is the number of replicates.
#'   - A list of \eqn{n_i}{ni}-vectors when locations vary across replicates. 
#'   - A list of the above elements (i.e., a list of matrices or a list of lists of vectors) when constructing graphs from multiple data sets.
#' @param isotropic Logical. If `TRUE`, edge features store the spatial distance (magnitude) 
#'   between nodes. If `FALSE`, the spatial displacement or spatial location is stored, depending 
#'   on the value of `stationary`.
#' @param stationary Logical. If `TRUE`, edge features store the spatial displacement 
#'   (vector difference) between nodes, capturing both magnitude and direction. If `FALSE`, 
#'   edge features include the full spatial locations of both nodes.
#' @param ... Additional keyword arguments from the Julia function [`adjacencymatrix()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/utility/#NeuralEstimators.adjacencymatrix) 
#' that define the neighborhood of each node, with the default being a randomly 
#' selected set of `k=30` neighbors within a radius of `r=0.15` spatial distance units.
#' @export 
#' @return A `GNNGraph` (`JuliaProxy` object) or, if multiple data sets are provided, a vector of `GNNGraph` objects which can be indexed in the usual manner using `[[` or converted to an R list using a combination of indexing and `lapply`. 
#' @examples
#' \dontrun{
#' library("NeuralEstimators")
#' 
#' # Number of replicates
#' m <- 5
#' 
#' # Spatial locations fixed for all replicates
#' n <- 100 
#' S <- matrix(runif(n * 2), n, 2)
#' Z <- matrix(runif(n * m), n, m)
#' g <- spatialgraph(S, Z)
#' 
#' # Spatial locations varying between replicates
#' n <- sample(50:100, m, replace = TRUE)
#' S <- lapply(n, function(ni) matrix(runif(ni * 2), ni, 2))
#' Z <- lapply(n, function(ni) runif(ni))
#' g <- spatialgraph(S, Z)
#' 
#' # Multiple data sets: Spatial locations fixed for all replicates within a given data set
#' K <- 15 # number of data sets
#' n <- sample(50:100, K, replace = TRUE) # number of spatial locations can vary between data sets
#' S <- lapply(1:K, function(k) matrix(runif(n[k] * 2), n[k], 2))
#' Z <- lapply(1:K, function(k) matrix(runif(n[k] * m), n[k], m))
#' g <- spatialgraph(S, Z)
#' 
#' # Multiple data sets: Spatial locations varying between replicates within a given data set
#' S <- lapply(1:K, function(k) {
#'   lapply(1:m, function(i) {
#'   ni <- sample(50:100, 1)       # randomly generate the number of locations for each replicate
#'   matrix(runif(ni * 2), ni, 2)  # generate the spatial locations
#'   })
#' })
#' Z <- lapply(1:K, function(k) {
#'   lapply(1:m, function(i) {
#'     n <- nrow(S[[k]][[i]])
#'     runif(n)  
#'   })
#' })
#' g <- spatialgraph(S, Z)
#' }
spatialgraph <- function(S, Z, isotropic = TRUE, stationary = TRUE, ...) {
  juliaEval('using NeuralEstimators')
  
  # Capture additional arguments from ...
  extra_args <- list(...)
  
  # Type check and conversion for `r` and `k` if provided
  if ("r" %in% names(extra_args)) {
    if (!is.numeric(extra_args$r)) {
      stop("`r` must be a numeric value (float).")
    }
    extra_args$r <- as.numeric(extra_args$r)
  }
  
  if ("k" %in% names(extra_args)) {
    if (!is.numeric(extra_args$k) || extra_args$k %% 1 != 0) {
      stop("`k` must be an integer value.")
    }
    extra_args$k <- as.integer(extra_args$k)
  }
  
  # Prepare Julia call string for keyword arguments
  extra_args_string <- paste0(
    names(extra_args), " = ", sapply(extra_args, function(arg) {
      if (is.character(arg)) paste0('"', arg, '"') else {
        if (!is.integer(arg) && is.numeric(arg) && arg %% 1 == 0) { # Ensure float is passed as float 
          return(paste0(arg, ".0"))
        } else {
          return(arg)
        }
      }
    }), collapse = ", "
  )
  
  # Determine if we have a single data set or multiple data sets 
  if (
    is.matrix(S) && is.matrix(Z) || 
    is.list(S) && all(sapply(S, is.matrix)) && is.list(Z) && all(sapply(Z, is.vector))
    ) {
    spatialgraph_call <- "spatialgraph"
  } else {
    spatialgraph_call <- "spatialgraph."
  }
    
  # Build Julia call dynamically to include additional arguments
  julia_call <- paste0(
    spatialgraph_call, 
    "(S, Z; stationary = stationary, isotropic = isotropic",
    if (length(extra_args) > 0) paste0(", ", extra_args_string) else "",
    ")"
  )
  
  # Execute the Julia call
  g <- juliaLet(julia_call, S = S, Z = Z, stationary = stationary, isotropic = isotropic)
  
  return(g)
}