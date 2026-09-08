# Coerce whole-number scalars to integers so R users can write 2 rather than 2L.
# Map the ASCII alias num_summaries_theta to Julia's num_summaries_theta (Greek theta).
.coerce_estimator_args <- function(dots) {
  nms <- names(dots)
  if (is.null(nms)) nms <- rep("", length(dots))
  if ("num_summaries_theta" %in% nms) {
    names(dots)[nms == "num_summaries_theta"] <- "num_summaries_\u03b8"
    nms <- names(dots)
  }
  integer_kwargs <- c("num_summaries", "num_summaries_\u03b8", "num_parameters")
  for (i in seq_along(dots)) {
    x <- dots[[i]]
    if (!is.numeric(x) || length(x) != 1L || is.na(x)) next
    if (x != suppressWarnings(as.integer(x))) next
    named_int <- nms[i] %in% integer_kwargs
    positional_int <- !nzchar(nms[i])
    if (named_int || positional_int) {
      dots[[i]] <- as.integer(x)
    }
  }
  dots
}

.call_estimator <- function(ctor, ...) {
  do.call(ctor, .coerce_estimator_args(list(...)))
}

#' @title PointEstimator
#'
#' @description Construct a neural Bayes point estimator. The neural-network
#' architecture is still defined in Julia (e.g. via \code{juliaEval()}); this
#' function wraps that architecture in a `PointEstimator`.
#'
#' Typical constructors, matching the Julia methods:
#' \itemize{
#' \item `PointEstimator(network)`: a single network mapping data to the parameter space.
#' \item `PointEstimator(num_parameters, summary_network, num_summaries = ...)`: a summary network with an MLP inference network built internally.
#' }
#'
#' @param ... arguments passed to the Julia version of [`PointEstimator()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/estimators#Bayes-estimators)
#' @return a `PointEstimator`
#' @export
#' @seealso [PosteriorEstimator()], [RatioEstimator()], [train()], [infer()]
#' @examples
#' \dontrun{
#' library("NeuralEstimators")
#' library("JuliaConnectoR")
#' juliaEval("using NeuralEstimators, Flux")
#' network <- juliaEval("
#'   n = 1; d = 2; w = 32
#'   psi = Chain(Dense(n, w, relu), Dense(w, w, relu))
#'   phi = Chain(Dense(w, w, relu), Dense(w, d))
#'   DeepSet(psi, phi)
#' ")
#' estimator <- PointEstimator(network)
#' }
PointEstimator <- function(...) {
  NE <- .getNeuralEstimators()
  .call_estimator(NE$PointEstimator, ...)
}

#' @title PosteriorEstimator
#'
#' @description Construct a neural posterior estimator. The neural-network
#' architecture is still defined in Julia (e.g. via \code{juliaEval()}); this
#' function wraps that architecture in a `PosteriorEstimator`.
#'
#' Typical constructors, matching the Julia methods:
#' \itemize{
#' \item `PosteriorEstimator(num_parameters, summary_network, num_summaries = ...)`: builds the approximate distribution internally (default `q = "NormalisingFlow"`).
#' \item `PosteriorEstimator(summary_network, q)`: an explicit approximate distribution `q`.
#' }
#'
#' The argument `q` may be a Julia type or a string naming one (e.g. `"Gaussian"`, `"GaussianMixture"`, `"NormalisingFlow"`).
#'
#' @param ... arguments passed to the Julia version of [`PosteriorEstimator()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/estimators#Posterior-estimators)
#' @return a `PosteriorEstimator`
#' @export
#' @seealso [PointEstimator()], [RatioEstimator()], [train()], [infer()]
PosteriorEstimator <- function(...) {
  NE <- .getNeuralEstimators()
  .call_estimator(NE$PosteriorEstimator, ...)
}

#' @title RatioEstimator
#'
#' @description Construct a neural likelihood-to-evidence-ratio estimator. The
#' neural-network architecture is still defined in Julia (e.g. via
#' \code{juliaEval()}); this function wraps that architecture in a `RatioEstimator`.
#'
#' Typical constructor, matching the Julia method:
#' \itemize{
#' \item `RatioEstimator(num_parameters, summary_network, num_summaries = num_summaries)`
#' }
#'
#' @param ... arguments passed to the Julia version of [`RatioEstimator()`](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/estimators#Ratio-estimators)
#' @return a `RatioEstimator`
#' @export
#' @seealso [PointEstimator()], [PosteriorEstimator()], [train()], [infer()]
RatioEstimator <- function(...) {
  NE <- .getNeuralEstimators()
  .call_estimator(NE$RatioEstimator, ...)
}
