# Instructions for creating/editing vignettes and uploading the package to CRAN

The vignettes for `NeuralEstimators` are built using `Rmd` files, which are converted into `html` documents using the vignette builder `knitr`. These `html` files are included as static vignettes on CRAN using the vignette builder `R.rsp`. Below are the steps to create and include vignettes in the package.

### Creating/editing vignettes

- In the `DESCRIPTION` set `VignetteBuilder: knitr`
- In `.Rbuildignore` uncomment `^vignettes/.*\.html$` and `^vignettes/.*\.html\.asis$` 
- In `.Rbuildignore` comment out `^vignettes/.*\.Rmd$`
- (Optional) In `.Rbuildignore` add `^vignettes/.*\.html$` and `^vignettes/.*\.html\.asis$` entries for the vignettes that you do not wish to update
- Create/edit the vignette `Rmd` file and place it in the folder `vignettes`
- Run `devtools::build_vignettes()`
- Move the generated `html` file from the folder `doc` to the folder `vignettes`, and create an `asis` file for the vignette if one is not already there

### Uploading the package to CRAN

- In the `DESCRIPTION` set `VignetteBuilder: R.rsp` and bump the version number if necessary
- In `.Rbuildignore` comment out `^vignettes/.*\.html$` and `^vignettes/.*\.html\.asis$` 
- In `.Rbuildignore` uncomment `^vignettes/.*\.Rmd$`
- Check the package using `devtools::check()`
- Build the package using `devtools::build()`
- Check the package using [WinBuilder](https://win-builder.r-project.org/)
- Once checked, submit to CRAN 