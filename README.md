# random-walk-lissajous

Taking a X-dimensional random walk and running a PCA will result in nice lissajous-figures when plotting them, see the python script. This is because of the underlying circulant structure of the random walk matrix (cumsum...).

The following papers explains this further in detail: (https://proceedings.neurips.cc/paper/2018/hash/7a576629fef88f3e636afd33b09e8289-Abstract.html): The paper also connects this concept to the training trajectories of ML models.

A few papers interrogate the structure of internal representations of concepts in LLMs (for example https://arxiv.org/abs/2602.15029), I think this may be related. So it may be a more fundamental mathmatical articact, rather than actual representational structures, yet this is only my intuition and not grounded in any reseach myself (yet). 
