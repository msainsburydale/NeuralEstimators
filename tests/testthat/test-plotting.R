set.seed(1)

test_that("packages can be loaded properly", {
  library("NeuralEstimators")
  expect_equal(1, 1)
})

test_that("plotrisk() is working", {
  # Generate toy data. Two estimators for a single parameter model,
  # sample sizes m = 1, 5, 10, 15, 25, and 30, and
  # 50 estimates for each combination of estimator and sample size:
  m         <- rep(rep(c(1, 5, 10, 15, 20, 25, 30), each = 50), times = 2)
  Z         <- lapply(m, rnorm)
  estimate  <- sapply(Z, mean)
  df <- data.frame(
    estimator = c("Estimator 1", "Estimator 2"),
    parameter = "mu", m = m, estimate = estimate, truth = 0
  )
  
  # Plot the risk function
  plotrisk(df)
  plotrisk(df, loss = function(x, y) (x-y)^2)
  
  expect_equal(1, 1)
})

test_that("plotdistribution() is working", {
  
  # In the following, we have two estimators and, for each parameter, 50 estimates
  # from each estimator.
  
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
  plotdistribution(df, parameter_labels = parameter_labels)
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
    ggplot() +
      geom_point(aes(x = x, y = y, shape = shape)) + 
      labs(shape = "") +
      theme_bw()
  })
  plotdistribution(df, parameter_labels = parameter_labels, type = "scatter", pairs = TRUE, upper_triangle_plots = upper_triangle_plots)
  
  expect_equal(1, 1)
})



