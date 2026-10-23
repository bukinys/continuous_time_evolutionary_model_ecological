# continuous_time_evolutionary_model_ecological
This repository contains an algorithm, implemented in Python, that models the evolutionary colonization of an ecosystem—characterized by a number of vacant ecological niches—by new species. The model is framed within the theory of stochastic processes and is described by a system of Kolmogorov differential equations for state probabilities.


speciation_model_eng.py - source code, with comments in English, for a model algorithm that uses the modified Euler method (predictor-corrector) to integrate the Kolmogorov system of equations for state probabilities.

speciation_model_rus.py - source code, with comments in Russian, for a model algorithm that uses Euler's method with refinement (predictor-corrector) to integrate the system of Kolmogorov equations for state probabilities.

Python version 3.3.6 or later is required.

To run the program, you need to install Python modules:
itertools
numpy
time
pandas

Running calculations from the command line:
python3 speciation_model_eng.py
or
python3 speciation_model_rus.py

To set the model parameters, see the comments within the source code.
