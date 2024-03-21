#' Plot estimates vs. true values. 
#'
#' @param df a long form data frame containing fields \code{estimator}, \code{parameter}, \code{estimate}, and \code{truth}.
#' @param parameter_labels a named vector containing parameter labels.
#' @param estimator_labels a named vector containing estimator labels.
#' @return a \code{'ggplot'} of the estimates for each parameter against the true value.
#' @export
#' @examples
#' \dontrun{
#' K <- 50
#' df <- data.frame(
#'   estimator = c("Estimator 1", "Estimator 2"), 
#'   parameter = rep(c("mu", "sigma"), each = K),
#'   truth = 1:(2*K), 
#'   estimate = 1:(2*K) + rnorm(4*K)
#' )
#' estimator_labels <- c("Estimator 1" = expression(hat(theta)[1]("·")),
#'                       "Estimator 2" = expression(hat(theta)[2]("·")))
#' parameter_labels <- c("mu" = expression(mu), "sigma" = expression(sigma))
#' 
#' plotestimates(df,  parameter_labels = parameter_labels, estimator_labels)}
plotestimates <- function(df, estimator_labels = waiver(), parameter_labels = NULL) {
  
  truth <- estimator <- NULL # Setting the variables to NULL first to appease CRAN checks (see https://stackoverflow.com/questions/9439256/how-can-i-handle-r-cmd-check-no-visible-binding-for-global-variable-notes-when)
  
  if (!is.data.frame(df)) df <- df$estimates # cater for the case that the user has passed in an "Assessment" object
  
  if (is.null(parameter_labels)) {
    param_labeller <- identity
  } else {
    param_labeller <- label_parsed
    df <- mutate_at(df, .vars = "parameter", .funs = factor, levels = names(parameter_labels), labels = parameter_labels)
  }
  
  ggplot(df) + 
    geom_point(aes(x=truth, y = estimate, colour  = estimator), alpha = 0.75) + 
    geom_abline(colour = "red", linetype = "dashed") +
    facet_wrap(~parameter, scales = "free", labeller = param_labeller, nrow = 1) +
    scale_colour_viridis(discrete = TRUE, labels = estimator_labels) +
    labs(colour = "") + 
    theme_bw() +
    theme(strip.background = element_blank())
}



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
#' \dontrun{
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
#' plotrisk(df, loss = function(x, y) (x-y)^2)}
plotrisk <- function(df, parameter_labels = NULL, loss = function(x, y) abs(x - y)) {
  
  # TODO add checks that df contains the correct columns
  
  truth <- estimator <- parameter <- m <- NULL # Setting the variables to NULL first to appease CRAN checks (see https://stackoverflow.com/questions/9439256/how-can-i-handle-r-cmd-check-no-visible-binding-for-global-variable-notes-when)

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

#' Plot the empirical sampling distribution of an estimator.
#'
#' @param df a long form data frame containing fields \code{estimator}, \code{parameter}, \code{estimate}, \code{truth}, and a column (e.g., \code{replicate}) to uniquely identify each observation.
#' @param type string indicating whether to plot kernel density estimates for each individual parameter (\code{type = "density"}) or scatter plots for all parameter pairs (\code{type = "scatter"}).
#' @param parameter_labels a named vector containing parameter labels.
#' @param estimator_labels a named vector containing estimator labels.
#' @param truth_colour the colour used to denote the true parameter value.
#' @param truth_size the size of the point used to denote the true parameter value (applicable only for \code{type = "scatter"}).
#' @param truth_line_size the size of the cross-hairs used to denote the true parameter value. If \code{NULL} (default), the cross-hairs are not plotted. (applicable only for \code{type = "scatter"}).
#' @param pairs logical; should we combine the scatter plots into a single pairs plot (applicable only for \code{type = "scatter"})?
#' @param upper_triangle_plots an optional list of plots to include in the uppertriangle of the pairs plot.
#' @param legend Flag; should we include the legend (only applies when constructing a pairs plot)
#' @param return_list Flag; should the parameters be split into a list?
#' @param flip Flag; should the boxplots be "flipped" using `coord_flip()` (default `FALSE`)?
#' @return a list of \code{'ggplot'} objects or, if \code{pairs = TRUE}, a single \code{'ggplot'}.
#' @export
#' @examples
#' \dontrun{
#' # In the following, we have two estimators and, for each parameter, 50 estimates
#' # from each estimator.
#' 
#' estimators <- c("Estimator 1", "Estimator 2")
#' estimator_labels <- c("Estimator 1" = expression(hat(theta)[1]("·")),
#'                       "Estimator 2" = expression(hat(theta)[2]("·")))
#'
#' # Single parameter:
#' df <- data.frame(
#'   estimator = estimators, truth = 0, parameter = "mu",
#'   estimate  = rnorm(2*50),
#'   replicate = rep(1:50, each = 2)
#' )
#' parameter_labels <- c("mu" = expression(mu))
#' plotdistribution(df)
#' plotdistribution(df, type = "density")
#' plotdistribution(df, parameter_labels = parameter_labels, estimator_labels = estimator_labels)
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
#' plotdistribution(df, parameter_labels = parameter_labels)
#' plotdistribution(df, parameter_labels = parameter_labels, type = "density")
#' plotdistribution(df, parameter_labels = parameter_labels, type = "scatter")
#' plotdistribution(df, parameter_labels = parameter_labels, type = "scatter", pairs = TRUE)
#'
#'
#' # Pairs plot with user-specified plots in the upper triangle:
#' upper_triangle_plots <- lapply(1:3, function(i) {
#'   x = rnorm(10)
#'   y = rnorm(10)
#'   shape = sample(c("Class 1", "Class 2"), 10, replace = TRUE)
#'   ggplot() +
#'     geom_point(aes(x = x, y = y, shape = shape)) + 
#'     labs(shape = "") +
#'     theme_bw()
#' })
#' plotdistribution(
#'     df, 
#'     parameter_labels = parameter_labels, estimator_labels = estimator_labels,
#'     type = "scatter", pairs = TRUE, upper_triangle_plots = upper_triangle_plots
#'     )}
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
  return_list = FALSE, 
  flip = FALSE
  ) {
  
  if (!is.data.frame(df)) df <- df$estimates # cater for the case that the user has passed in an "Assessment" object

  type <- match.arg(type)
  if(!is.logical(pairs)) stop("pairs should be logical")
  if(!is.logical(flip)) stop("flip should be logical")
  if (!all(c("estimator", "parameter", "estimate", "truth") %in% names(df))) stop("df must contain the fields estimator, parameter, estimate, and truth.")
  if ("k" %in% names(df) && length(unique(df$k)) > 1) stop("df contains a column 'k' which has more than one unique value; this means that you are trying to visualise the distribution for more than one parameter configuration. To do this, please split the data frame by k and then use lapply() to generate a list of plots grouped by k.")
  if ("m" %in% names(df) && length(unique(df$m)) > 1) stop("df contains a column 'm' which has more than one unique value; this means that you are trying to visualise the distribution for more than one sample size. To do this, please split the data frame by m and then use lapply() to generate a list of plots grouped by m.") #TODO just average over m? 
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
      gg <- .marginalplotlist(df, parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_colour = truth_colour, type = type, flip = flip)
    } else {
      gg <- .marginalplot(df, parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_colour = truth_colour, type = type, flip = flip)
    }
  } else if (type == "scatter") {
    gg <- .scatterplot(df, parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_colour = truth_colour, truth_size = truth_size, truth_line_size = truth_line_size)
    if (pairs) {
      gg <- .pairsplot(gg, parameter_labels = parameter_labels, upper_triangle_plots = upper_triangle_plots, legend = legend)
    }
  }

  return(gg)
}

aes_string_quiet <- function(...) suppressWarnings(aes_string(...))

.scatterplot <- function(df, parameter_labels, truth_colour, estimator_labels, truth_size, truth_line_size) {
  
  parameter <- NULL # Setting the variables to NULL first to appease CRAN checks (see https://stackoverflow.com/questions/9439256/how-can-i-handle-r-cmd-check-no-visible-binding-for-global-variable-notes-when)

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
        aes_string_quiet(
          x = paste("estimate", p[1], sep = "_"),
          y = paste("estimate", p[2], sep = "_"),
          colour = "estimator"
          ),
        alpha = 0.75) +
      geom_point(
        aes_string_quiet(
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
        geom_vline(aes_string_quiet(xintercept = paste("truth", p[1], sep = "_")), colour = truth_colour, size = truth_line_size) +
        geom_hline(aes_string_quiet(yintercept = paste("truth", p[2], sep = "_")), colour = truth_colour, size = truth_line_size)
    }

    return(gg)
  })

  return(scatterplots)
}

.marginalplot <- function(df, parameter_labels, truth_colour, type, estimator_labels, flip) {
  
  estimator <- truth <- NULL # Setting the variables to NULL first to appease CRAN checks

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
    if (flip) {
      gg <- gg +
        coord_flip() + 
        theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
        scale_x_discrete(limits=rev) + 
        theme(axis.title.y = element_blank())
    } else {
      gg <- gg + theme(
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank()
      ) 
    }
  }

  return(gg)
}

# plotdistribution(df, parameter_labels = parameter_labels, return_list = TRUE)
# plotdistribution(df, parameter_labels = parameter_labels, flip = TRUE)

.marginalplotlist <- function(df, parameter_labels, truth_colour, type, estimator_labels, flip) {
  
  parameter <- estimator <- truth <- NULL # Setting the variables to NULL first to appease CRAN checks

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
      gg <- gg + labs(x = parameter_labels[param])
      if (flip) {
        gg <- gg +
          coord_flip() + 
          theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
          scale_x_discrete(limits=rev)
      } else {
        gg <- gg +
          theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) 
      }

    } else if (type == "density") {
      gg <- gg +
        labs(title = parameter_labels[param]) +
        theme(plot.title = element_text(hjust = 0.5))
    }

    gg
  })
}


.pairsplot <- function(scatterplots, parameter_labels, upper_triangle_plots, legend = legend) {

  p <- length(parameter_labels)

  scatterplots <- lapply(scatterplots, function(gg) gg + theme(axis.title = element_blank()))

  # Extract legend so that it can be placed in the final plot
  scatterplot_legend.grob <- get_legend(scatterplots[[1]])
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
  upper_idx <- (choose(p + 1, 2) + 1):p^2
  layout[upper.tri(layout)] <- upper_idx
  if (is.null(upper_triangle_plots)) {
    upper_triangle_plots <- lapply(upper_idx, function(i) ggplot() + theme_void())
    upper_triangle_plots_legend_grob <- NULL
  } else {
    upper_triangle_plots_legend_grob <<- get_legend(upper_triangle_plots[[1]])
    if (length(upper_triangle_plots) != choose(p, 2)) stop("The number of upper_triangle_plots should be choose(p, 2), where p is the number of parameters in the statistical model")
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
  legend_width <- 0.5 #NB could add an argument for the relative width of the legends
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