% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/missingdata.R
\name{encodedata}
\alias{encodedata}
\title{encodedata}
\usage{
encodedata(Z, c = 0)
}
\arguments{
\item{Z}{data containing \code{NA} entries}

\item{c}{fixed constant with which to replace \code{NA} entries}
}
\value{
Augmented data set (U, W). If \code{Z} is provided as a list, the return type will be a \code{JuliaProxy} object; these objects can be indexed in the usual manner using \code{[[}, or converted to an R object using \code{juliaGet()} (note however that \code{juliaGet()} can be slow for large data sets).
}
\description{
For data \code{Z} with missing (\code{NA}) entries, computes an augmented data set (U, W) where W encodes the missingness pattern as an indicator vector and U is the original data Z with missing entries replaced by a fixed constant \code{c}.
}
\examples{
\dontrun{
library("NeuralEstimators")
Z <- matrix(c(1, 2, NA, NA, 5, 6, 7, NA, 9), nrow = 3)
encodedata(Z)
encodedata(list(Z, Z))}
}
\seealso{
the Julia version of \href{https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/utility/#NeuralEstimators.encodedata}{\code{encodedata()}}
}
