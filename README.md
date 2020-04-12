# ANN-MIC
MIC implementation of an ANN (artificial neural network)
The ANN algorithm is learning model inspired by biological neurons of human brain. The Multi-Layer Perceptron (MLP) is an ANN 
in which each node has avalue that propagates to the nodes of the next layer. Every node of one layer is fully connected to 
all nodes of the adjacent layer. 
The MLP is determined by the number of nodes and the number of layers an the weights between each layer nodes. 
The input values propagate to the next layer resulted by the weighted sum of the input nodes and an activation function (sigmoid). 
This calulation approximates the unknown function with only a statistical learning model.
GPU Processing.

Operations on each layer have data parallelism so that they can be mapped on a GPU architecture.
The output vector Y is the output values for nodes where wij is the weight of the i-th front node to the j-th back node.
And xij is the input value of the i-th front node.
N is the number of front nodes and M is the number of back nodes.

Y =( f(w11*x1+w21*x1+ ...+wN1*x1 + b1), ... f(w1M*xM+w2M*xM+ ...+wNM*xM + bM) )
Each row of Y (the inner produucts) can be mapped to individual threads. These threads share the data and
they need to communicate for summation. The sum needs to be mapped in shares memory because thread communication in
other memories is too slow in GPUs.

Threads cannot communicate through shared memory if they are not in the same thread block.

Accelerating with GPUs
ANN need a large amount of training data to reach a high accuracy in their predictions.
ANN parameters must be adjusted through back-propagation and therefore they need
a large amount of training data: hundreds of thousands to millions of input samples
have to be run thropugh forward and backward passes.
GPUs can provide a signficant computation speedup with respect to training ANN with CPU only.
GPUs, tehrefore, have become the standard platform for training large and complex
ANN-based systems because of their ability to accelerate these systems.
NVIDIA has a library of primitives called cuDNN that makes it easy to obtain performance with Deep NN.
NVIDIA has also  an inference platform accelerator and runtime called TensorRT. 
TensorRT delivers low-latency, high-throughput inference and tunes the runtime application 
to run optimally across different families of GPUs.
