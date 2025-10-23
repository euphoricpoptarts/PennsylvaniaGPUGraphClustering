# Using Objective Functions in Our Clustering Programs

## Overview of LambdaCC Objective Functions

Many of the most common objective functions for graph clustering can be expressed by the following equation.

```math
\lambda CC(C) = \sum_{C_i \in C} \sum_{u,v \in C_i} (w(u,v) - \lambda w(u)w(v))
```

This equation is monotonic in objectives such as Modularity, the Constant-Potts Model, and even connected components.  
With proper choice of vertex and edge weights, these and other objectives can be expressed in one convenient model.

## Objective Function Options
**Mod**: Use the Modularity objective. $\lambda = \frac{1}{w(E)}$, where $w(E)$ encapsulates the common factor of 2 seen in modularity objectives.
Vertex weights are given by their degree. Edge weights have unit weight, which will override any given edge weights in the graph file.
Not compatible with custom vertex weights.
This is the default objective if you do not specify one.  
**WMod**: Use the Weighted Modularity objective. $\lambda = \frac{1}{w(E)}$. Differs from modularity in that edge weights are not automatically overriden to 1. Vertex weights are given by the sum of adjacent edge weights.
Not compatible with custom vertex weights.  
**CPM**: Use the Constant-Potts Model. Vertex and edges are given unit weight. $\lambda = \frac{1}{2}$.
Not compatible with custom vertex weights.  
**NLCC**: Use the Normalized LambdaCC objective. $\lambda = \frac{w(E)}{w(V)^2}$. Does not override edge weights.
Compatible with custom vertex weights.
Uses unit vertex weights if custom vertex weights are not given.
**LCC**: Use the Base LambdaCC objective. $\lambda = 1$. Does not override edge weights.
Compatible with custom vertex weights.
Uses unit vertex weights if custom vertex weights are not given.  

## Lambda
The **-lambda** parameter multiplies the $\lambda$ used by each objective function.
It is effectively the same as the resolution parameter commonly used with Modularity and the Constant-Potts Model.
It is not limited to these objectives, as you can use it to rescale the $\lambda$ of every objective.

## Custom Vertex Weights
If you wish to use the Normalized LambdaCC or Base LambdaCC objectives, you can specify a file with custom vertex weights.  
This file should have one integer per line, with $|V|$ lines. The weight for vertex $x$ should appear on line $x$.