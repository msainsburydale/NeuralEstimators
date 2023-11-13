# ---- plotrisk() ----

#' Plot the risk function (with respect to a given loss function) versus the sample size, m.
#'
#' @param df a long form data frame containing fields \code{estimator},
#' \code{parameter}, \code{estimate}, \code{truth}, and \code{m}.
#' @param parameter_labels a named vector containing parameter labels used in the plot.
#' @param loss the loss function; defaults to absolute-error loss.
#' @return a \code{'ggplot'} object showing the risk function versus the sample size, facetted by parameter.
#' @export
#' @seealso \code{\link{plotdistribution}}
#' @examples
#' # Generate toy data. Two estimators for a single parameter model,
#' # sample sizes m = 1, 5, 10, 15, 25, and 30, and
#' # 50 estimates for each combination of estimator and sample size:
#' m         <- rep(rep(c(1, 5, 10, 15, 20, 25, 30), each = 50), times = 2)
#' Z         <- lapply(m, rnorm)
#' estimate  <- sapply(Z, mean)
#' df <- data.frame(
#'   estimator = c("Estimator 1", "Estimator 2"),
#'   parameter = "mu", m = m, estimate = estimate, truth = 0
#' )
#'
#' # Plot the risk function
#' plotrisk(df)
#' plotrisk(df, loss = function(x, y) (x-y)^2)
plotrisk <- function(df, parameter_labels = NULL, loss = function(x, y) abs(x - y)) {

  df <- df %>% mutate(residual = estimate - truth)

  if (is.null(parameter_labels)) {
    param_labeller <- identity
  } else {
    param_labeller <- label_parsed
    df <- mutate_at(df, .vars = "parameter", .funs = factor, levels = names(parameter_labels), labels = parameter_labels)
  }

  # Compute global risk for each combination of estimator and sample size m
  df <- df %>%
    group_by(estimator, parameter, m) %>%
    dplyr::summarise(loss = mean(loss(estimate, truth)))

  # Plot risk vs. m
  gg <- ggplot(data = df, aes(x = m, y = loss, colour = estimator, group = estimator)) +
    geom_point() +
    geom_line() +
    facet_wrap(parameter ~ ., scales = "free", labeller = param_labeller) +
    labs(colour = "", x = expression(m), y = expression(r[Omega](hat(theta)))) +
    theme_bw() +
    theme(legend.text.align = 0,
          panel.grid = element_blank(),
          strip.background = element_blank())

  return(gg)
}

# ---- plotdistribution() ----

#' Plot the empirical distribution of several estimators.
#'
#' @param df a long form data frame containing fields \code{estimator}, \code{parameter}, \code{estimate}, \code{truth}, and a column (e.g., \code{replicate}) to uniquely identify each observation.
#' @param type string indicating whether to plot kernel density estimates for each individual parameter (\code{type = "density"}) or scatter plots for all parameter pairs (\code{type = "scatter"}).
#' @param parameter_labels a named vector containing parameter labels used in the plot.
#' @param truth_colour the colour used to denote the true parameter value.
#' @param truth_size the size of the point used to denote the true parameter value (applicable only for \code{type = "scatter"}).
#' @param truth_line_size the size of the cross-hairs used to denote the true parameter value. If \code{NULL} (default), the cross-hairs are not plotted. (applicable only for \code{type = "scatter"}).
#' @param pairs logical; should we combine the scatter plots into a single pairs plot (applicable only for \code{type = "scatter"})?
#' @param upper_triangle_plots an optional list of plots to include in the uppertriangle of the pairs plot.
#' @param legend Flag; should we include the legend (only applies when constructing a pairs plot)
#' @param return_list Flag; should the parameters be split into a list?
#' @return a list of \code{'ggplot'} objects or, if \code{pairs = TRUE}, a single \code{'ggplot'}.
#' @export
#' @examples
#' # In the following, we have two estimators and, for each parameter, 50 estimates
#' # from each estimator.
#'
#' # Single parameter:
#' estimators <- c("Estimator 1", "Estimator 2")
#' df <- data.frame(
#'   estimator = estimators, truth = 0, parameter = "mu",
#'   estimate  = rnorm(2*50),
#'   replicate = rep(1:50, each = 2)
#' )
#'
#' parameter_labels <- c("mu" = expression(mu))
#' estimator_labels <- c("Estimator 1" = expression(hat(theta)[1]("·")),
#'                       "Estimator 2" = expression(hat(theta)[2]("·")))
#'
#' plotdistribution(df, parameter_labels = parameter_labels, estimator_labels = estimator_labels)
#' plotdistribution(df, parameter_labels = parameter_labels, type = "density")
#'
#'
#' # Two parameters:
#' df <- rbind(df, data.frame(
#'   estimator = estimators, truth = 1, parameter = "sigma",
#'   estimate  = rgamma(2*50, shape = 1, rate = 1),
#'   replicate = rep(1:50, each = 2)
#' ))
#' parameter_labels <- c(parameter_labels, "sigma" = expression(sigma))
#' plotdistribution(df, parameter_labels = parameter_labels)
#' plotdistribution(df, parameter_labels = parameter_labels, type = "density")
#' plotdistribution(df, parameter_labels = parameter_labels, type = "scatter")
#'
#'
#' # Three parameters:
#' df <- rbind(df, data.frame(
#'   estimator = estimators, truth = 0.25, parameter = "alpha",
#'   estimate  = 0.5 * runif(2*50),
#'   replicate = rep(1:50, each = 2)
#' ))
#' parameter_labels <- c(parameter_labels, "alpha" = expression(alpha))
#' plotdistribution(df, parameter_labels = parameter_labels)
#' plotdistribution(df, parameter_labels = parameter_labels, type = "density")
#' plotdistribution(df, parameter_labels = parameter_labels, type = "scatter")
#' plotdistribution(df, parameter_labels = parameter_labels, type = "scatter", pairs = TRUE)
#' plotdistribution(df, parameter_labels = parameter_labels, type = "scatter", pairs = TRUE, legend = FALSE)
#'
#'
#' # Pairs plot with user-specified plots in the upper triangle:
#' upper_triangle_plots <- lapply(1:3, function(i) {
#'   x = rnorm(10)
#'   y = rnorm(10)
#'   shape = sample(c("Class 1", "Class 2"), 10, replace = TRUE)
#'   qplot(x = x, y = y, shape = shape) +
#'     labs(shape = "") +
#'     theme_bw()
#' })
#' plotdistribution(df, parameter_labels = parameter_labels, type = "scatter", pairs = TRUE, upper_triangle_plots = upper_triangle_plots)
plotdistribution <- function(
  df,
  type = c( "box", "density", "scatter"),
  parameter_labels = NULL,
  estimator_labels = waiver(),
  truth_colour = "red",
  truth_size = 8,
  truth_line_size = NULL,
  pairs = FALSE,
  upper_triangle_plots = NULL,
  legend = TRUE,
  return_list = FALSE
  ) {

  type <- match.arg(type)
  if(!is.logical(pairs)) stop("pairs should be logical")
  if (!all(c("estimator", "parameter", "estimate", "truth") %in% names(df))) stop("df must contain the fields estimator, parameter, estimate, and truth.")
  if ("k" %in% names(df) && length(unique(df$k)) > 1) stop("df contains a column 'k' which has more than one unique value; this means that you are trying to visualise the distribution for more than one parameter configuration. To do this, please split the data frame by k and then use lapply() to generate a list of plots grouped by k.")
  if ("m" %in% names(df) && length(unique(df$m)) > 1) stop("df contains a column 'm' which has more than one unique value; this means that you are trying to visualise the distribution for more than one sample size. To do this, please split the data frame by m and then use lapply() to generate a list of plots grouped by m.")
  if (!is.null(upper_triangle_plots) && !pairs) warning("The argument upper_triangle_plots is ignored when pairs == FALSE")

  if(is.null(parameter_labels)) {
    parameter_labels <- unique(df$parameter)
    names(parameter_labels) <- parameter_labels
  }
  p <- length(parameter_labels)
  param_names <- unique(df$parameter)
  if (!all(param_names %in% names(parameter_labels))) stop("Some parameters have not been given parameter labels: Please ensure all(unique(df$parameter) %in% names(parameter_labels))")
  if (p != length(param_names)) {
    parameter_labels <- parameter_labels[param_names]
  }

  if (p == 1 && type == "scatter") {
    warning("Setting type = 'density' since the number of parameters is equal to 1.")
    type = "density"
  }

  if (p < 3 && pairs) {
    warning("Setting pairs = FALSE since the number of parameters is less than 3.")
    pairs = FALSE
  }

  if (type == "box" | type == "density") {
    if (return_list) {
      gg <- .marginalplotlist(df, parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_colour = truth_colour, type = type)
    } else {
      gg <- .marginalplot(df, parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_colour = truth_colour, type = type)
    }
  } else if (type == "scatter") {
    gg <- .scatterplot(df, parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_colour = truth_colour, truth_size = truth_size, truth_line_size = truth_line_size)
    if (pairs) {
    gg <- .pairsplot(gg, parameter_labels = parameter_labels, upper_triangle_plots = upper_triangle_plots, legend = legend)
    }
  }

  return(gg)
}


.scatterplot <- function(df, parameter_labels, truth_colour, estimator_labels, truth_size, truth_line_size) {

  # all parameter pairs
  combinations <- parameter_labels %>% names %>% combinat::combn(2) %>% as.matrix

  # convert to wide form
  df <- df %>%
    pivot_wider(names_from = parameter, values_from = c("estimate", "truth")) %>%
    as.data.frame

  # Generate the scatterplot estimation panels
  scatterplots <- apply(combinations, 2, function(p) {

    gg <- ggplot(data = df[sample(nrow(df)), ]) +
      geom_point(
        aes_string(
          x = paste("estimate", p[1], sep = "_"),
          y = paste("estimate", p[2], sep = "_"),
          colour = "estimator"
          ),
        alpha = 0.75) +
      geom_point(
        aes_string(
          x = paste("truth", p[1], sep = "_"),
          y = paste("truth", p[2], sep = "_")
          ),
        colour = truth_colour, shape = "+", size = truth_size
        ) +
      labs(colour = "", x = parameter_labels[[p[1]]], y = parameter_labels[[p[2]]]) +
      scale_colour_viridis(discrete = TRUE, labels = estimator_labels) +
      theme_bw()

    if (!is.null(truth_line_size)) {
      gg <- gg +
        geom_vline(aes_string(xintercept = paste("truth", p[1], sep = "_")), colour = truth_colour, size = truth_line_size) +
        geom_hline(aes_string(yintercept = paste("truth", p[2], sep = "_")), colour = truth_colour, size = truth_line_size)
    }

    return(gg)
  })

  return(scatterplots)
}

.marginalplot <- function(df, parameter_labels, truth_colour, type, estimator_labels) {

  if (is.null(parameter_labels)) {
    param_labeller <- identity
  } else {
    param_labeller <- label_parsed
    df <- mutate_at(df, .vars = "parameter", .funs = factor, levels = names(parameter_labels), labels = parameter_labels)
  }

  gg <- ggplot(df)

  if (type == "box") {
    gg <- gg +
      geom_boxplot(aes(y = estimate, x = estimator, colour = estimator)) +
      geom_hline(aes(yintercept = truth), colour = truth_colour, linetype = "dashed")
  } else if (type == "density"){
    gg <- gg +
      geom_line(aes(x = estimate, group = estimator, colour = estimator), stat = "density") +
      geom_vline(aes(xintercept = truth), colour = truth_colour, linetype = "dashed")
  }

  gg <- gg +
    facet_wrap(parameter ~ ., scales = "free", labeller = param_labeller) +
    labs(colour = "") +
    scale_colour_viridis(discrete = TRUE, labels = estimator_labels) +
    theme_bw() +
    theme(
      legend.text.align = 0,
      panel.grid = element_blank(),
      strip.background = element_blank()
      )

  if (type == "box") {
    gg <- gg + theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.title.x = element_blank()
      )
  }

  return(gg)
}

# plotdistribution(df, parameter_labels = parameter_labels, type = "box", return_list = TRUE)
# plotdistribution(df, parameter_labels = parameter_labels, type = "density", return_list = TRUE)

.marginalplotlist <- function(df, parameter_labels, truth_colour, type, estimator_labels) {

  parameters <- names(parameter_labels)

  lapply(parameters, function(param) {

    df <- df %>% filter(parameter == param)
    gg <- ggplot(df)

    if (type == "box") {
      gg <- gg +
        geom_boxplot(aes(y = estimate, x = estimator, colour = estimator)) +
        geom_hline(aes(yintercept = truth), colour = truth_colour, linetype = "dashed")
    } else if (type == "density"){
      gg <- gg +
        geom_line(aes(x = estimate, group = estimator, colour = estimator), stat = "density") +
        geom_vline(aes(xintercept = truth), colour = truth_colour, linetype = "dashed")
    }

    gg <- gg +
      labs(colour = "") +
      scale_colour_viridis(discrete = TRUE, labels = estimator_labels) +
      theme_bw() +
      theme(
        legend.text.align = 0,
        panel.grid = element_blank(),
        strip.background = element_blank()
      )

    if (type == "box") {
      gg <- gg +
        labs(x = parameter_labels[param]) +
        theme(axis.text.x = element_blank(),
              axis.ticks.x = element_blank())
    } else if (type == "density") {
      gg <- gg +
        labs(title = parameter_labels[param]) +
        theme(plot.title = element_text(hjust = 0.5))
    }

    gg
  })
}



# ---- plotdistribution(): pairsplot ----


.pairsplot <- function(scatterplots, parameter_labels, upper_triangle_plots, legend = legend) {

  # TODO Need to add a check that the number of upper_triangle_plots is ok.
  # could just recycle to correct length with a warning.

  # TODO Decide if we want estimator_labels (I think it makes sense to include it,
  # since it's difficult for the user to edit the legend of pairs plots.
  # Sanity check:
  # if (!all(network %in% names(estimator_labels))) stop("Not all estimators have been given a label")

  p <- length(parameter_labels)

  scatterplots <- lapply(scatterplots, function(gg) gg + theme(axis.title = element_blank()))

  # Extract legend so that it can be placed in the final plot
  scatterplot_legend.grob <<- get_legend(scatterplots[[1]])
  scatterplots <- lapply(scatterplots, function(gg) gg + theme(legend.position = "none"))

  param_names <- names(parameter_labels)
  layout      <- matrix(rep(NA, p^2), nrow = p)

  # Lower-diagonal part of the plot: Parameter estimates
  layout[lower.tri(layout)] <- 1:choose(p, 2)
  plotlist <- scatterplots

  # Diagonal part of the plot (parameter names)
  diag(layout) <- (choose(p, 2) + 1):(choose(p, 2) + p)
  diag_plots <- lapply(seq_along(parameter_labels), function(i) {
    ggplot() +
      annotate("text", x = 0, y = 0, label = parameter_labels[i], size = 8) +
      theme_void()
  })
  plotlist <- c(plotlist, diag_plots)

  # Upper-diagonal part of the plot
  upper_idx <- (choose(p, 2) + p + 1):p^2
  layout[upper.tri(layout)] <- upper_idx
  if (is.null(upper_triangle_plots)) {
    upper_triangle_plots <- lapply(upper_idx, function(i) ggplot() + theme_void())
    upper_triangle_plots_legend_grob <- NULL
  } else {
    upper_triangle_plots_legend_grob <<- get_legend(upper_triangle_plots[[1]])
    upper_triangle_plots <- lapply(upper_triangle_plots, function(gg) gg + theme(legend.position = "none"))
  }
  plotlist <- c(plotlist, upper_triangle_plots)



  # Add the scatterplots legend
  if (legend) {
    legend_part <- rep(p^2 + 1, p)
    layout   <- cbind(legend_part, layout)
    plotlist <- c(plotlist, list(scatterplot_legend.grob))
  }

  if (!is.null(upper_triangle_plots_legend_grob)) {
    # Add upper_triangle_plots legend
    legend_part <- rep(p^2 + 2, p)
    layout <- cbind(layout, legend_part)
    plotlist <- c(plotlist, list(upper_triangle_plots_legend_grob))
  }

  # Put everything together
  legend_width <- 0.5 #TODO could add an argument for the relative width of the legends
  suppressWarnings(
    gg <- grid.arrange(
      grobs = plotlist, layout_matrix = layout,
      widths = c(
        if(legend) legend_width,
        rep(1, p),
        if(!is.null(upper_triangle_plots_legend_grob)) legend_width
        )
      )
  )

  # Convert to object of class ggplot for consistency with other return values
  gg <- ggplotify::as.ggplot(gg)

  return(gg)
}


# ---- plottrainingrisk() ----

#' Plots the evolution of the risk function during training.
#'
#' @param path a path containing one or more folders titled "runs_x", each of
#' which corresponds to a training run of a neural estimator and, hence, each
#' contains a .csv file called 'loss_per_epoch.csv'.
#' @param excluded_runs folders to exclude (e.g., "runs_N1" to exclude the folder "runs_N1").
#' @return a \code{'ggplot'} showing the evolution of the risk function during training for each neural estimator.
#' @export
#' @seealso \code{\link{plotrisk}}, \code{\link{plotdistribution}}
plottrainingrisk <- function(path, excluded_runs = NULL) {

  # Find the runs_ folders:
  all_dirs <- list.dirs(path = path, recursive = TRUE)
  runs_dirs <- all_dirs[which(grepl("runs_", all_dirs))]
  if (!is.null(excluded_m)) runs_dirs <- runs_dirs[!grepl(excluded_m, runs_dirs)]

  # Load the loss function per epoch files:
  loss_per_epoch_list <- lapply(runs_dirs, function(x) {
    loss_per_epoch <- read.csv(paste0(x, "/loss_per_epoch.csv"), header = FALSE)
    colnames(loss_per_epoch) <- c("training", "validation")
    loss_per_epoch$epoch <- 0:(nrow(loss_per_epoch) - 1)
    return(loss_per_epoch)
  })

  # Extract the title of each network:
  network <- sub(".*runs_", "", runs_dirs)

  # Extract the number of replicates used during training for each estimator:
  m <- regmatches(network, gregexpr("[[:digit:]]+", network)) %>% as.numeric

  # TODO Decide if we want estimator_labels (I think it makes sense to include it,
  # since it's difficult for the user to edit the legend of pairs plots.
  # Sanity check:
  # if (!all(network %in% names(estimator_labels))) stop("Not all estimators have been given a label")

  # Combine the loss matrices into a single data frame:
  df <- do.call("rbind", loss_per_epoch_list)
  df$network <- rep(network, sapply(loss_per_epoch_list, nrow))
  df$m       <- rep(m, sapply(loss_per_epoch_list, nrow))
  df <- df %>%
    melt(c("epoch", "network", "m"), variable.name = "set", value.name = "loss")# %>%
  #mutate_at(.vars = "network", .funs = factor, labels = estimator_labels[network])

  # Create y limits using the minimum loss over all sets for a given m, so that
  # the panels are directly comparable when appropriate.
  # (see here https://stackoverflow.com/a/42590452)
  df <- df %>%
    group_by(m) %>%
    mutate(ymin = min(loss), ymax = max(loss))

  # Compute the minimum validation loss for a given m.
  min_df <- df %>% filter(set == "validation") %>% summarise(min_val_loss = min(loss))
  min_val_loss <- setNames(min_df$min_val_loss, min_df$m)
  df$min_val_loss <- min_val_loss[as.character(df$m)]

  # Plot the loss functions:
  gg <- ggplot(df) +
    geom_line(aes(x = epoch, y = loss, colour = set)) +
    scale_color_manual(values = c("blue", "red")) +
    facet_wrap(~network, scales = "free", labeller = label_parsed, nrow = 1) +
    labs(colour = "", y = "loss") +
    geom_hline(aes(yintercept = min_val_loss), colour = "red", alpha = 0.3, linetype = "dashed") +
    geom_blank(aes(y = ymin)) +
    geom_blank(aes(y = ymax)) +
    theme_bw()  +
    theme(
      panel.grid = element_blank(),
      strip.background = element_blank()
    )

  return(gg)
}


remove_geom <- function(ggplot2_object, geom_type) {
  # Delete layers that match the requested type.
  layers <- lapply(ggplot2_object$layers, function(x) {
    if (class(x$geom)[1] == geom_type) {
      NULL
    } else {
      x
    }
  })
  # Delete the unwanted layers.
  layers <- layers[!sapply(layers, is.null)]
  ggplot2_object$layers <- layers
  ggplot2_object
}