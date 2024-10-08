% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/core.R
\name{assess}
\alias{assess}
\title{assess a neural estimator}
\usage{
assess(
  estimators,
  parameters,
  Z,
  estimator_names = NULL,
  parameter_names = NULL,
  use_gpu = TRUE,
  verbose = TRUE
)
}
\arguments{
\item{estimators}{a list of (neural) estimators}

\item{parameters}{true parameters, stored as a pxK matrix, where p is the number of parameters in the statistical model and K is the number of sampled parameter vectors}

\item{Z}{data simulated conditionally on the \code{parameters}. If \code{Z} contains more data sets than parameter vectors, the parameter matrix will be recycled by horizontal concatenation.}

\item{estimator_names}{list of names of the estimators (sensible defaults provided)}

\item{parameter_names}{list of names of the parameters (sensible defaults provided)}

\item{use_gpu}{a boolean indicating whether to use the GPU if it is available (default true)}

\item{verbose}{a boolean indicating whether information should be printed to the console}
}
\value{
a list of two data frames: \code{runtimes}, contains the
total time taken for each estimator, while \code{estimates} is a long-form
data frame with columns:
\itemize{
\item{"estimator"; the name of the estimator}
\item{"parameter"; the name of the parameter}
\item{"truth"; the true value of the parameter}
\item{"estimate"; the estimated value of the parameter}
\item{"m"; the sample size (number of iid replicates)}
\item{"k"; the index of the parameter vector in the test set}
\item{"j"; the index of the data set}
}
}
\description{
assess a neural estimator
}
\seealso{
\code{\link[=risk]{risk()}}, \code{\link[=rmse]{rmse()}}, \code{\link[=bias]{bias()}}, \code{\link[=plotestimates]{plotestimates()}}, and \code{\link[=plotdistribution]{plotdistribution()}} for computing various empirical diagnostics and visualisations based on an assessment object
}
