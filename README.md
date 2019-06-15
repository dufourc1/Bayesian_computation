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

Beware: at the time of writing, there is a compatibility issue between autograd and scipy that can be easily fixed following [this issue](https://github.com/HIPS/autograd/issues/501) from the github page of autograd.

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
         -Gaussian_exp_prior(Prior) 
                 Implementation of the gaussian prior for the parameters beta and the
                exponential prior for the sigma: beta ~ N(mean,std**2), sigma ~ Exp(lamdba)
         -Gaussian_gamma_prior(Prior)
                 Implementation of the Student prior for the parameters beta and the
                 gamma prior for the df of the student noise: beta ~ N(mean,std**2), df ~ Gamma(alpha,beta)
         -Gaussian_prior(Prior):
                 Implementation of the vanilla gaussian prior: theta ~ N(mean,std**2)
         
`prior.log_prior`
`prior.log_prior_grad`
`prior.log_prior_hessian`

    Conditional_model
        - Gaussian linear model
        - Student linear model
        - Logistic model 
        - Gamma model (not checked for errors)
       
`cond_mod.log_l`
`cond_mod.log_l_grad`
`cond_mod.log_l_hessian`

See this [issue](https://github.com/dufourc1/Bayesian_computation/issues/2) for the gamma model

### optimization

  Implementation of the optimization routine seen in the course, specialized to treat the problem
  of maximizing the posterior of a model by minimizing the negative log posterior of the same model

    vanilla_gd

    line_search_gd

    Wolfe_cond_gd

    stochastic_gd 

    newton_gd (unstable)
    
See this  [issue](https://github.com/dufourc1/Bayesian_computation/issues/4)


### approximation

  Implementation of various approximation techniques:

  these methods take as argument a model and return and approximation of its normalized posterior

    GVA (not entirely checked for errors) 
    
    laplace_approx
 See [issue](https://github.com/dufourc1/Bayesian_computation/issues/3)


### sampling

  Implementation of the sampling routines:

  these methods take as argument a model and return samples from its unormalized posterior

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
