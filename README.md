<div align="center">
  <img src="https://github.com/fmfn/axia/blob/master/examples/axia-logo.png"><br><br>
</div>

-----------------

The axia package provides an implementation of an extension  of
the Shifted-Beta-Geometric model by P. Fader & B. Hardie [1] to individuals in
contractual settings with multiple predictor variables.

It will soon provide a lot more...

Axia offers a survival analysis like approach to modeling the
behaviour of individuals in contractual settings. Given a set of features, ages
and status (alive or dead), this package builds on top of the model developed
by P. Fader & B. Hardie [1], to infer hazard curves, retention curves and
discounted-life-time-value.

Head over to the examples folder to learn how to use axia.

Installation
======
Axia is not currently available on PyPi. To install the package,
you will need to clone it and run the setup.py file. Use the following commands to
get a copy from Github and install all dependencies:

    git clone https://github.com/fmfn/axia.git
    cd axia
    python setup.py install
### Dependencies
* numpy
* pandas
* scipy
* jupyter (for examples only)

References
===
1. ["HOW TO PROJECT CUSTOMER RETENTION", P. Fader & B. Hardie](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf)
2. ["Customer-Base Valudation in a Contractual Setting: The Perils of Ignoring
Heterogeneity", P. Fader & B. Hardie](https://pdfs.semanticscholar.org/d620/c9a463b9d09d433e01ecec4db083bb4a2ac9.pdf?_ga=2.242218690.1394399945.1550257837-1536596951.1529000020)
