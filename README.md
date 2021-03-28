# Build-a-Recurrent-Neural-Network
Recurrent Neural Networks (RNN) are very effective for Natural Language Processing and other sequence tasks because they have "memory". They can read inputs $x^{\langle t \rangle}$ (such as words) one at a time, and remember some information/context through the hidden layer activations that get passed from one time-step to the next. This allows a uni-directional RNN to take information from the past to process later inputs. A bidirection RNN can take context from both the past and the future.

Notation:

Superscript $[l]$ denotes an object associated with the $l^{th}$ layer.

Example: $a^{[4]}$ is the $4^{th}$ layer activation. $W^{[5]}$ and $b^{[5]}$ are the $5^{th}$ layer parameters.
Superscript $(i)$ denotes an object associated with the $i^{th}$ example.

Example: $x^{(i)}$ is the $i^{th}$ training example input.
Superscript $\langle t \rangle$ denotes an object at the $t^{th}$ time-step.

Example: $x^{\langle t \rangle}$ is the input x at the $t^{th}$ time-step. $x^{(i)\langle t \rangle}$ is the input at the $t^{th}$ timestep of example $i$.
Lowerscript $i$ denotes the $i^{th}$ entry of a vector.

Example: $a^{[l]}_i$ denotes the $i^{th}$ entry of the activations in layer $l$.
