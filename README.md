# Kronecker Dual Layers (KDL)

A pair of layers based on Kronecker poduct multiplication designed to replace a "large" dense layer and reduce computational time.

## Description

A dense layer with weight matrix $W$ and input vector $\mathbf{x}$ requires the product $W\mathbf{x}$.   A sum of Kronecker products may be used to form the 
representation $W = \displaystyle\sum_{i=1}^r R_i^T \otimes L_i$, where $r$ is the Kronecker rank.  THis is beneficial since the product may now be written as 
$W\mathbf{x} = \text{vec}\left(\displaystyle\sum_{i=1}^r L_i X R_i \right)$, where $\text{mat}\left(\mathbf{x}\right) = X$ and $\text{mat}()$ and $\text{vec}()$ 
are matricization and vectorization functions repectively for appropriate dimensions.  However, to produce an accurate representation, $r$ may be large.  Thus, 
the layer is "split" into dual layers, each with separate activation and bias terms, to increase predictive capabilities while reducing the required Kronecker 
rank.  The order in which the layers are implemented doesn't appear to effect the training accuracy or timing, since $L(XR) = (LX)R$ before adding additional 
bias and activation terms are added.  A full description is available in the ArXiv preprint: 
[Dimensionality Reduction in Deep Learning via Kronecker Multi-layer Architectures](https://arxiv.org/abs/2204.04273) by Jarom D. Hogue, Robert M. Kriby and Akil Narayan.

## Code

Included are three files:

* KDLayers.py: includes custom Tensorflow layers KDLeftLayer and KDRightLayer.
* KDL_mnist_example: standalone example python notebook comparing KDL neural net (rank 1 and 2) with an FNN for the MNIST dataset.
* KDL_cifar10_example: standalone example python notenook comparing CNN's using KDL's (rank 1 and 4) with a CNN using dense layers for the cifar10 dataset.

## Contributor

* Jarom D. Hogue: [jdhogue@sci.utah.edu](mailto:jdhogue@sci.utah.edu)

## Liscense

This project is available for use under the MIT license, see [license](/LICENSE.md).
