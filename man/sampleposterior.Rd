% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/core.R
\name{sampleposterior}
\alias{sampleposterior}
\title{sampleposterior}
\usage{
sampleposterior(
  estimator,
  Z,
  theta_grid,
  N = 1000,
  prior = NULL,
  use_gpu = TRUE
)
}
\arguments{
\item{estimator}{a neural likelihood-to-evidence-ratio estimator}

\item{Z}{data; it's format should be amenable to the architecture of \code{estimator}}

\item{theta_grid}{a (fine) gridding of the parameter space, given as a matrix with p rows, where p is the number of parameters in the model}

\item{N}{number of samples to draw (default 1000)}

\item{prior}{the prior (default uniform), specified as a Julia or R function}

\item{use_gpu}{a boolean indicating whether to use the GPU if it is available (default true)}
}
\value{
a p × \code{N} matrix of posterior samples, where p is the number of parameters in the model. If multiple data sets are given in \code{Z}, a list of posterior samples will be returned
}
\description{
Given data \code{Z}, a neural likelihood-to-evidence-ratio \code{estimator}, and a \code{prior}, draws samples from the implied approximate posterior distribution

Currently, the sampling algorithm is based on a fine-gridding \code{theta_grid} of the parameter space. The approximate posterior density is evaluated over this grid, which is then used to draw samples. This is very effective when making inference with a small number of parameters. For models with a large number of parameters, other sampling algorithms may be needed (please feel free to contact the package maintainer for discussion).
}
\seealso{
\code{\link[=mlestimate]{mlestimate()}}, \code{\link[=mapestimate]{mapestimate()}}
}
