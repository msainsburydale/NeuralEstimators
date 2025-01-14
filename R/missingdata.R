#' @title encodedata
#'
#' @description for data `Z` with missing (`NA`) entries, returns an augmented data set (U, W) where W encodes the missingness pattern as an indicator vector and U is the original data Z with missing entries replaced by a fixed constant `c`.
#' 
#' The indicator vector W is stored in the second-to-last dimension of `Z`, which should be singleton. If the second-to-last dimension is not singleton, then two singleton dimensions will be added to the array, and W will be stored in the new second-to-last dimension.
#'
#'
#' @param Z data containing `NA` entries
#' @param c fixed constant with which to replace `NA` entries
#' @return Augmented data set (U, W). If `Z` is provided as a list, the return type will be a `JuliaProxy` object; these objects can be indexed in the usual manner (e.g., using `[[`), or converted to an R object using `juliaGet()` (note however that `juliaGet()` can be slow for large data sets). 
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


#' @title spatialgraph
#'
#' @description Constructs a graph object for use in graph neural networks. 
#'
#' @param S Spatial locations, provided as:
#'   - An `n x d` matrix when locations are fixed across replicates, where `n` is the number 
#'     of spatial locations and `d` is the spatial dimension (typically `d=2`).
#'   - A list of `n_i x d` matrices when locations vary across replicates, where each matrix 
#'     corresponds to the spatial locations of a single replicate.
#' @param Z Spatial data, provided as:
#'   - An `n x m` matrix when locations are fixed, where `m` is the number of replicates.
#'   - A list of `n_i` vectors when locations vary across replicates, where each vector 
#'     corresponds to the data for a single replicate.
#' @param isotropic Logical. If `TRUE`, edge features store the spatial distance (magnitude) 
#'   between nodes. If `FALSE`, the spatial displacement or spatial location is stored, depending 
#'   on the value of `stationary`.
#' @param stationary Logical. If `TRUE`, edge features store the spatial displacement 
#'   (vector difference) between nodes, capturing both magnitude and direction. If `FALSE`, 
#'   edge features include the full spatial locations of both nodes.
#' @param ... Additional keyword arguments from the Julia function [`adjacencymatrix()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/utility/#NeuralEstimators.adjacencymatrix) 
#' determine the neighborhood of each node, with the default being a randomly 
#' selected set of `k=30` neighbors within a radius of `r=0.15` units.
#'   
#' @seealso [spatialgraphlist()] for the vectorised version of this function.
#' 
#' @export 
#' 
#' @return A `GNNGraph` (a `JuliaProxy` object) ready for use in a graph neural network.
#'
#' @details
#' 
#'
#' @examples
#' \dontrun{
#' library("NeuralEstimators")
#' 
#' # Number of replicates and spatial dimension
#' m <- 5
#' d <- 2
#' 
#' # Spatial locations fixed for all replicates
#' n <- 100
#' S <- matrix(runif(n * d), n, d)
#' Z <- matrix(runif(n * m), n, m)
#' g <- spatialgraph(S, Z)
#' 
#' # Spatial locations varying between replicates
#' n <- sample(50:100, m, replace = TRUE)
#' S <- lapply(n, function(ni) matrix(runif(ni * d), ni, d))
#' Z <- lapply(n, function(ni) runif(ni))
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
  
  # Build Julia call dynamically to include additional arguments
  julia_call <- paste0(
    "spatialgraph(S, Z; stationary = stationary, isotropic = isotropic",
    if (length(extra_args) > 0) paste0(", ", extra_args_string) else "",
    ")"
  )
  
  # Execute the Julia call
  g <- juliaLet(julia_call, S = S, Z = Z, stationary = stationary, isotropic = isotropic)
  return(g)
}

#' @title spatialgraphlist
#'
#' @description Constructs a list of graph objects for use in a graph neural network. 
#' 
#' @param S A list of spatial locations, where each element of the list is: 
#'   - An `n x d` matrix when locations are fixed across replicates in a given data set, where `n` is the number 
#'     of spatial locations and `d` is the spatial dimension (typically `d=2`).
#'   - A list of `n_i x d` matrices when locations vary across replicates, where each matrix 
#'     corresponds to the spatial locations of a single replicate.
#' @param Z A list of spatial data, where each element of the list is: 
#'   - An `n x m` matrix when locations are fixed, where `m` is the number of replicates.
#'   - A list of `n_i` vectors when locations vary across replicates, where each vector 
#'     corresponds to the data for a single replicate.
#' @inheritParams spatialgraph
#' 
#' @seealso [spatialgraph()] for the non-vectorised version of this function.
#' 
#' @export 
#' 
#' @return A list of `GNNGraph` objects stored as a `JuliaProxy` object; this object can be indexed in the usual manner (e.g., using `[[`), and converted to an R list using a combination of indexing and `lapply`. 
#'
#' @details
#' Additional keyword arguments from the Julia function 
#' [`adjacencymatrix()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/utility/#NeuralEstimators.adjacencymatrix) 
#' determine the neighborhood of each node, with the default being a randomly selected set of `k=30` neighbors within a radius of `r=0.15` units.
#' 
#' @examples
#' \dontrun{
#' library("NeuralEstimators")
#' 
#' # Number of data sets, number of replicates in each data set, and spatial dimension
#' K <- 15
#' m <- 5
#' d <- 2
#' 
#' # Spatial locations fixed for all replicates within a given data set
#' n <- 100
#' S <- lapply(1:K, function(k) matrix(runif(n * d), n, d))
#' Z <- lapply(1:K, function(k) runif(n))
#' g <- spatialgraphlist(S, Z)
#' 
#' # Spatial locations varying between replicates within a given data set
#' S <- lapply(1:K, function(k) {
#'   lapply(1:m, function(i) {
#'     ni <- sample(50:100, 1)       # Randomly generate the number of locations for each replicate
#'     matrix(runif(ni * d), ni, d)  # Generate the spatial locations
#'   })
#' })
#' Z <- lapply(1:K, function(k) {
#'   lapply(1:m, function(i) {
#'     n <- nrow(S[[k]][[i]])
#'     runif(n)  
#'   })
#' })
#' g <- spatialgraphlist(S, Z)
#' }
spatialgraphlist <- function(S, Z, isotropic = TRUE, stationary = TRUE, ...) {
  
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
  
  # Build Julia call dynamically to include additional arguments
  julia_call <- paste0(
    "spatialgraph.(S, Z; stationary = stationary, isotropic = isotropic",
    if (length(extra_args) > 0) paste0(", ", extra_args_string) else "",
    ")"
  )
  
  # Execute the Julia call
  g <- juliaLet(julia_call, S = S, Z = Z, stationary = stationary, isotropic = isotropic)
  
  return(g)
}