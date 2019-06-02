# Bayesian computation framework from scratch

Project for the course of Bayesian computation (MATH-435).

The purpose of this project was to write the low levels functions and algorithms used in bayesian
statistics and to use them on a simple dataset.

## Libraries used
The following libraries were used for this project, with Python 3.6.8


 Computational:

    numpy    (1.16.4)
    scipy    (1.3.0)
    autograd (1.2)
    pandas   (0.24.2)


Graphical:

    matplotlib  (3.1.0)
    seaborn     (0.9.0)



## Prerequisites



The folder structure is the following:

    ├── data                  #csv of the dataset
    ├── results                              
        ├── plots
        └── csv
    ├── src                    # Source files (actual framework)
        ├── helpers.py
        ├── models                           
        ├── optimization
        ├── approximation
        └── sampling
    ├── project.ipynb          #project interface and use on dataset
    └── README.md


## Element of the source files

### models

Contains the following classes:

    Model:

        A model consist of a prior and a conditional model plus the data corresponding
        to the problem. Using autograd, one can compute the posterior, the log posterior,
        the gradient and the hessian of those quantities directly by calling:
`model.log_posterior`
`model.log_posterior_grad`
`model.log_posterior_hessian`

    Prior
`prior.log_prior`
`prior.log_prior_grad`
`prior.log_prior_hessian`

    Conditional_model
`cond_mod.log_l`
`cond_mod.log_l_grad`
`v.log_l_hessian

### optimization

  Implementation of the optimization routine seen in the course, specialized to treat the problem
  of maximizing the posterior of a model by minimizing the negative log posterior of the same model

    vanilla_gd

    line_search_gd

    Wolfe_cond_gd

    stochastic_gd (notImplemented)

    newton_gd (notImplemented)



### approximation

  Implementation of various approximation techniques:

    GVA

    laplace_approx

### sampling

  Implementation of the sampling routines

    importance_sampling

    rejection_sampling

    Gibbs_sampling (notImplemented)

    random_walk_MH

    Langevin_MH


## Project.iynb

The details of the statistical analysis are written in the notebook, only the data is presented here.

### Dataset
 Wages in the USA dataset (May 1985) from  _E. R. Berndt. The practice of econometric : classic and contemporary. Addison-Wesley Pub. Co., 1991_

 The dataset contains the following data:

- Continous variables


   | ED 	| EX 	| LNWAGE|
   |-------|--------| -----|
   | education in years |  experience in years   | the logarithm of the hourly wage  |

- Categorical variables about the type of employment


   | MANAG 	| SALES 	| CLER | SERV | PROF |
   |-------|--------| -----| --| ----|
   |managerial |  sales worker   | clerical worker  | service worker | technical worker |

- Categorical data about the industry

    | MANUF 	| CONSTR 	|
    |-------|--------|
    | manufacturing |   construction  |

- Other Categorical data:

     | FE 	| MARR	| MARRFE| SOUTH | NONWH | HISP |
     |-------|--------| -----|---|---|---|
     | sex (1 if female) | marital status    | marital status of female  | geograpical data| non white | hispanique|

## Author

* *Charles Dufour*
