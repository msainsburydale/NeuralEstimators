# Instructions for creating/updating vignettes and uploading the package to CRAN

The vignettes for `NeuralEstimators` are built using `Rmd` files, which are converted into `html` documents using the vignette builder `knitr`. These `html` files are included as static vignettes on CRAN using the vignette builder `R.rsp`. Below are the steps to create and include vignettes in the package:

### Creating/updating vignettes

- In the `DESCRIPTION` set `VignetteBuilder: knitr`
- In `.Rbuildignore` uncomment `^vignettes/.*\.html$` and `^vignettes/.*\.html\.asis$` 
- In `.Rbuildignore` comment out `^vignettes/.*\.Rmd$`
- Create the vignette `Rmd` file and place it in the folder `vignettes`
- Run `devtools::build_vignettes()`

### Uploading the package to CRAN

- In the `DESCRIPTION` set `VignetteBuilder: R.rsp` and bump the version number
- In `.Rbuildignore` comment out `^vignettes/.*\.html$` and `^vignettes/.*\.html\.asis$` 
- In `.Rbuildignore` uncomment `^vignettes/.*\.Rmd$`
- Build the package using `devtools::build()`
- Check the package using `devtools::check()`
- Check the package on Windows using [WinBuilder](https://win-builder.r-project.org/)
- Once checked, submit the package to CRAN 