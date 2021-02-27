#CoNSEPT
A Convolutional Neural Network-based Sequence-to-Expression Prediction Tool. This repo contains the CoNSEPT model we used to model the expression of rhomboid gene (rho) in Drosophila. This version has more or less the same architecture as the [main version](https://github.com/PayamDiba/CoNSEPT) but it can be trained only on the particular data set we provided in ```data``` (data obtained from [here](https://elifesciences.org/articles/08445)).


## Usage
For general purpose usages see the [main version](https://github.com/PayamDiba/CoNSEPT) providing more detailed explanations for all model arguments.

Here is an example command-line to train the model provided in the current repo:


```python data/train.py --sf data/seq.fa --ef data/expr.tab --tf data/factor_expr.tab --pwm data/factors.wtmx --nb 17 --nTrain 38 --nValid 3 --nTest 11 --ne 52 --psb 10,2 --csc 10,2 --sc 10,2 --nChans 6,3 --cAct relu --oAct sigmoid --dr 0.5 --nEpoch 1000 --bs 20 --o out_dir/example_model --restore True```
