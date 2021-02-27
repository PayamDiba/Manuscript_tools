# GEMSTAT
Saurabh Sinha’s Lab, University of Illinois at Urbana-Champaign [Sinha Lab](https://www.sinhalab.net/sinha-s-home)

## Description
A thermodynamics-based model of transcriptional regulation by enhancers. This version of GEMSTAT supports limited-contact scheme of activation.

## Authors:
Originally developed by Xin He <xinhe2@illinois.edu>

Some features added and code was refactored by Bryan Lunt <bjlunt2@illinois.edu>

Limited-contact direct activation scheme was added by Payam Dibaeinia <dibaein2@illinois.edu>

## Installation
The program needs GSL (GNU Scientific Library). After installing GSL, go to the main distribution directory, and type:
```
./configure
make
```

The main executable (the main program to fit a sequence-to-expression model) will be generated at ```src/seq2expr```. Type the program name without parameters will print basic usage information.

## Usage
The program takes the following arguments as input:

* -s: the sequence file in FASTA format, required.  

* -e: the expression data of the sequences. The first line specifies the name of the expression conditions, required.  

* -m: the PWM (motif) of the relevant TFs, required.  

* -f: the expression data of the TFs. Must match the format of expr_file, and the order of TFs in motif_file, required.  

* -fo: path to the output directory. Predicted expression is written here, required.  

* -o: the repression mechanism to be used. All model use direct activation mechanism. Options are: Direct (Direct repression with unlimited-contact for direct activation), Direct_Limited (Direct repression with limited-contact for direct activation), ChrMod_Unlimited (Neighborhood remodeling repression with unlimited-contact for direct activation), ChrMod_Limited (Neighborhood remodeling repression with limited-contact for direct activation).  

* -c: the list of cooperative/quenching interactions. One line per interacting pair. When set the third column to "SIMPLE", the fourth column specifies the distance threshold of the interaction.  If not specified, then no interaction is allowed.  

* -i: the role of TFs (activators or repressors). The second column indicates whether the TF is an activator and the third whether repressor (in theory, an activator could have two roles, thus we have two columns).  

* -oo: the option of objective function. Options are: SSE - sum of squared error (default), Corr - average correlation coefficient.  

* -mc: the max-contact number for "Limited" contact models.  

* -p: the parameter file. When this option is specified, the values of the parameters in par_file will be used as initial parameter values of the optimizer.  

* -na: the number of alternations between two optimization methods (GSL simplex method and GSL BFGS method). If it is 0, no parameter estimation. Typically 3 to 5.  

* -a: the file of sequences represented by a set of TFBSs (site annotation). With this option, only the specified sites will be used in the sequence-to-expression model. Note that in the file, the first column is the start position of a site (from 1), the second is the strand of the site, and the third is the factor name.  

* -ct: the distance threshold of cooperative interactions. Default = 50 bp.  

* -rt: the distance threshold of short range repression in ChrMod_Unlimited and ChrMod_Unlimited. Default = 150 bp.

* -lower_bound: the lower bounds for all free parameters. See ```example\lower.par``` for an example of such file.  

* -upper_bound: the upper bounds for all free parameters. See ```example\upper.par``` for an example of such file.  

* -ff: a file that determines free and fixed (to initial value) parameters during training. See ```example\free_fix.par``` for an example of such file.  

* -onebeta: whether use the same scaling for all input enhancers (true), or use different scalings (false).


## Example
We provided an example dataset in ```example``` directory (data obtained from [here](https://elifesciences.org/articles/08445)). The data consists of expressions driven by 38 enhancers in 17 trans conditions regulated by three TFs. Here is an example command-line for training a GEMSTAT model with neighborhood remodeling short-range repression, limited-contact direct activation and cooperative interactions between four pairs of TFs.

{PATH-TO}/src/seq2expr -s {PATH-TO}/example/seq.fa -e {PATH-TO}/example/expr.tab -m {PATH-TO}/example/factors.wtmx -f {PATH-TO}/example/factor_expr.tab -i {PATH-TO}/example/factor_info.txt -c {PATH-TO}/example/coop.txt -lower_bound {PATH-TO}/example/lower.par -upper_bound {PATH-TO}/example/upper.par -a {PATH-TO}/example/annotations.txt -o ChrMod_Limited -mc 1 -rt 100 -ff {PATH-TO}/example/free_fix.txt -onebeta true -oo SSE -na 5 -p {PATH-TO}/example/par.txt -fo {PATH-TO}/output/predictions.txt


In order to train an ensemble of models, prepare an ensemble of initial parameters and run the above command once for each parameter setting using ```-p``` flag.
