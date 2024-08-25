testthat::skip_on_cran()

set.seed(1)

test_that("packages can be loaded properly", {
  library("NeuralEstimators")
  library("JuliaConnectoR")
  expect_equal(1, 1)
})

test_that("julia can be called", {
  x <- juliaEval('
  1 + 1
')
  expect_equal(x, 2)
})

test_that("Flux is available", {
  juliaEval('
  # Install the package if not already installed
  using Pkg
  installed = "Flux" ∈ keys(Pkg.project().dependencies)
  if !installed
    Pkg.add("Flux")  
  end
  using Flux
')
  expect_equal(1, 1)
})

test_that("the Julia version of NeuralEstimators is available", {
  juliaEval('
  # Install the package if not already installed
  using Pkg
  installed = "NeuralEstimators" ∈ keys(Pkg.project().dependencies)
  if !installed
    Pkg.add(url = "https://github.com/msainsburydale/NeuralEstimators.jl") 
  end
  using NeuralEstimators
')
  expect_equal(1, 1)
})


test_that("plotestimates() is working", {
  K <- 50
  df <- data.frame(
    estimator = c("Estimator 1", "Estimator 2"), 
    parameter = rep(c("mu", "sigma"), each = K),
    truth = 1:(2*K), 
    estimate = 1:(2*K) + rnorm(4*K)
  )
  estimator_labels <- c("Estimator 1" = expression(hat(theta)[1]("·")),
                        "Estimator 2" = expression(hat(theta)[2]("·")))
  parameter_labels <- c("mu" = expression(mu), "sigma" = expression(sigma))
  plotestimates(df,  parameter_labels = parameter_labels, estimator_labels)
  
  expect_equal(1, 1)
})

test_that("plotdistribution() is working", {
  
  # Single parameter:
  estimators <- c("Estimator 1", "Estimator 2")
  df <- data.frame(
    estimator = estimators, truth = 0, parameter = "mu",
    estimate  = rnorm(2*50),
    replicate = rep(1:50, each = 2)
  )
  
  parameter_labels <- c("mu" = expression(mu))
  estimator_labels <- c("Estimator 1" = expression(hat(theta)[1]("·")),
                        "Estimator 2" = expression(hat(theta)[2]("·")))
  
  plotdistribution(df, parameter_labels = parameter_labels, estimator_labels = estimator_labels)
  plotdistribution(df, parameter_labels = parameter_labels, type = "density")
  
  # Two parameters:
  df <- rbind(df, data.frame(
    estimator = estimators, truth = 1, parameter = "sigma",
    estimate  = rgamma(2*50, shape = 1, rate = 1),
    replicate = rep(1:50, each = 2)
  ))
  parameter_labels <- c(parameter_labels, "sigma" = expression(sigma))
  plotdistribution(df, return_list = TRUE)
  plotdistribution(df, parameter_labels = parameter_labels)
  plotdistribution(df, parameter_labels = parameter_labels, flip = TRUE)
  plotdistribution(df, parameter_labels = parameter_labels, flip = TRUE, return_list = TRUE)
  plotdistribution(df, parameter_labels = parameter_labels, type = "density")
  plotdistribution(df, parameter_labels = parameter_labels, type = "scatter")
  
  # Three parameters:
  df <- rbind(df, data.frame(
    estimator = estimators, truth = 0.25, parameter = "alpha",
    estimate  = 0.5 * runif(2*50),
    replicate = rep(1:50, each = 2)
  ))
  parameter_labels <- c(parameter_labels, "alpha" = expression(alpha))
  plotdistribution(df, parameter_labels = parameter_labels)
  plotdistribution(df, parameter_labels = parameter_labels, type = "density")
  plotdistribution(df, parameter_labels = parameter_labels, type = "scatter")
  plotdistribution(df, parameter_labels = parameter_labels, type = "scatter", pairs = TRUE)
  plotdistribution(df, parameter_labels = parameter_labels, type = "scatter", pairs = TRUE, legend = FALSE)
  
  # Pairs plot with user-specified plots in the upper triangle:
  upper_triangle_plots <- lapply(1:3, function(i) {
    x = rnorm(10)
    y = rnorm(10)
    shape = sample(c("Class 1", "Class 2"), 10, replace = TRUE)
    ggplot2::ggplot() +
      ggplot2::geom_point(ggplot2::aes(x = x, y = y, shape = shape)) + 
      ggplot2::labs(shape = "") +
      ggplot2::theme_bw()
  })
  plotdistribution(df, parameter_labels = parameter_labels, type = "scatter", pairs = TRUE, upper_triangle_plots = upper_triangle_plots)
  
  
  expect_equal(1, 1)
})