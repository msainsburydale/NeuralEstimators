removeoutliers <- function(df, Ω, scaling_factor = 2) {

  # Expand the support by scaling_factor
  if (scaling_factor < 1) stop("scaling_factor should be greater than 1")
  scaling_factor = c(1/scaling_factor, scaling_factor)
  Ω = lapply(Ω, function(x) x * scaling_factor)

  # First, add a two-column matrix to the data frame; the ith row of this matrix
  # specifies the bounds for the parameter in the ith row of df.
  bounds <- do.call(rbind, Ω[df$parameter])
  bounds <- as.data.frame(bounds)
  names(bounds) <- c("lower", "upper")
  rownames(bounds) <- NULL
  df <- cbind(df, bounds)

  # Now find which estimates are outside of the bounds (i.e., which are outliers):
  outliers <- which(with(df, lower > estimate | estimate > upper))
  cat("Number of likelihood-based estimates that are outliers:", length(outliers), "\n")

  # Find and remove parameter configurations associated with these outliers.
  # We remove entire pairs of parameter configuration and sample size even if
  # only one of the parameters is considered an outlier. This is because the
  # failure of one parameter would likely compromise the estimation of the
  # others.
  bad_mk <- df[outliers, c("m", "k")]
  df <- anti_join(df, bad_mk, by = c("m", "k"))

  # remove lower and upper
  df$lower <- NULL
  df$upper <- NULL

  df
}

#' @title repeat parameters in the form expected by NeuralEstimators.jl
#' @export
repeatparameters <- function(parameters, num_rep) parameters[, rep(1:ncol(parameters), each = num_rep), drop = FALSE]
