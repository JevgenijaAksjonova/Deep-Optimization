Accelerated Forward-Backward Optimization using Deep Learning
==================================

This repository contains the code for the article "[Accelerated Forward-Backward Optimization using Deep Learning](https://arxiv.org/abs/2105.05210)".

Contents
--------
The code contains the following

* Training using anthropomorphic data from Mayo Clinic.
* Evaluation on example slice
* Implementation of the baseline methods.

Dependencies
------------
The code depends on tensorflow-gpu=1.15.0, [ODL](https://github.com/odlgroup/odl) and [ASTRA toolbox](https://www.astra-toolbox.com/). 
ODL and ASTRA can be installed by 

```bash
$ git clone https://github.com/odlgroup/odl
$ cd odl
$ pip install --editable .
$ conda install -c astra-toolbox/label/dev astra-toolbox
```

Contact
-------
Jevgenija Rudzusika,  
KTH, Royal Institute of Technology   
jevaks@kth.se

Sebastian Banert,  
Lund University,  
sebastian.banert@control.lth.se

Jonas Adler,  
DeepMind,  
jonasadler@google.com

Ozan Öktem,  
KTH, Royal Institute of Technology  
ozan@kth.se

Funding
-------
Development is financially supported by the Swedish Foundation of StrategicResearch grant AM13-0049, grant from the VINNOVA Open Innovation Hubproject 2015-06759, and by Philips Healthcare.
