The Fuzzy temporal Hamming distance
===================================

From the paper : 

* A Fuzzy Generalisation of the Hamming Distance for Temporal Sequences
> C Moreau, T Devogele, C de Runz, V Peralta, E Moreau, L Etienne
> Fuzz-IEEE, 2021

--

## About parameters:

### Beta variable
-------------

Beta variable controls the frontiers of the fuzzy membership function. 

Beta -> ∞ <=> All symbols in sequences are taken into account in sequences. 

Beta -> 0 <=> Hamming distance


### Sim function
------------

The sim:Σ x Σ -> [0,1] function defined the similarity between all symbols in the alphabet of sequences Σ. 
Basicaly, we can use the trival distance function. 
The Wu-Palmer similarity function used a knowledge graph (i.e., ontology) for symbol comparison. An example of graph structure is given in the file "ontology_sac.txt". 


### Interval_step
------------

Granularity of the time interval I according to the defined unit of time. 



