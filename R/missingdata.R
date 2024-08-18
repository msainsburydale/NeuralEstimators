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
