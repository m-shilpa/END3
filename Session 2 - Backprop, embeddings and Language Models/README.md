# Backpropagation in a Feed Forward Network

Feedforward neural networks were among the first and most successful learning algorithms. They are also called deep networks, multi-layer perceptron (MLP), or simply neural networks. As data travels through the network’s artificial mesh, each layer processes an aspect of the data, filters outliers, spots familiar entities and produces the final output.

Feedforward neural networks are made up of the following:

1. **Input layer:** This layer consists of the neurons that receive inputs and pass them on to the other layers. The number of neurons in the input layer should be equal to the attributes or features in the dataset.
2. **Output layer:** The output layer is the predicted feature and depends on the type of model you’re building.
Hidden layer: In between the input and output layer, there are hidden layers based on the type of model. Hidden layers contain a vast number of neurons which apply transformations to the inputs before passing them. As the network is trained, the weights are updated to be more predictive. 
3. **Neuron weights:** Weights refer to the strength or amplitude of a connection between two neurons. If you are familiar with linear regression, you can compare weights on inputs like coefficients. Weights are often initialized to small random values, such as values in the range 0 to 1.

Below is a simple 2 layered Feed Forward Network :

![image](https://github.com/m-shilpa/END3/blob/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/feedforward_nn.png)

We will use this network as reference for our calculations.

## Training a Feed Forward Network 

The Feed Forward Network takes the following steps for training:

### Step 1: Forward Propagation

In the Forward Propagation, as the name suggests, the input data is fed in the forward direction through the network. Each hidden layer accepts the input data, processes it as per the activation function and passes to the successive layer.

At each neuron in a hidden or output layer, the processing happens in two steps:

1. **Preactivation:** It is a weighted sum of inputs i.e. the linear transformation of weights w.r.t to inputs available. Based on this aggregated sum and activation function the neuron makes a decision whether to pass this information further or not. This can be represented as below:

    ![image](https://github.com/m-shilpa/END3/blob/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/neuron_eq.png)

    So, each of the neurons shown in the above neural network can be calculated as follows:
    ```
    h1 = w1 * i1 + w2 * i2				

    h2 = w3 * i1 + w4 * i2				

    o1 = a_h1 * w5 + a_h2 * w6

    o2 = a_h1 * w7 + a_h2 * w8	
    ```

2. **Activation:** The calculated weighted sum of inputs is passed to the activation function. Activation functions bring non-linearity into the network. Without the use of activation functions, our neural network is nothing but a linear function acting as a single layer even if it is a 1000 layers neural network. There are four commonly used and popular activation functions — sigmoid, hyperbolic tangent(tanh), ReLU and Softmax. In our example, we will be using the sigmoid function as our activation function.
The sigmoid function is represented as below:

![image](https://github.com/m-shilpa/END3/blob/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/sigmoid_eqn.png)

Following are the equations for the neurons:
```
a_h1 = σ(h1)

a_h2 = σ(h2)

a_o1 = σ(o1)

a_o2 = σ(o2)
```

### Step 2: Error

Error tells us how far our model predictions are from the actual output. The simplest way to find this is as follows:

Error = 1/2( actual output - model predictions )²

Here we are using the squared error.

For the neural network shown above, the errors, E1 and E2 are as follows:
```
E1 = 1/2 (t1 - a_o1)²

E2 = 1/2 (t2 - a_o2)²
```

The total Error E_T is :
```
E_T = E1 + E2
```

We need to minimize this error E_T. To minimise this error we need to change the values which were involved in arriving at this error. These are the input, weights and bias. The input is fixed and hence cannot be changed. So, to minimize the error, we should optimise our model parameters which are the weights and bias. This is performed during backpropagation.

### Step 3: Backpropagation

Backpropagation is the essence of neural network training. It is the method of fine-tuning the weights of a neural network based on the error rate obtained in the previous epoch (i.e., iteration). Proper tuning of the weights allows you to reduce error rates and make the model reliable by increasing its generalization.

Each of the weights are updated based on the following :
```
w = w - LR (∂e/ ∂w)
```
LR stands for learning rate. It is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient.

From the above equation, we see that, we need to calculate the derivative of error with respect to each of the weights in the neural network.

Why do we need to use derivative ?

Derivative tells us, by how much the error changes when the weight changes each time. In other words,derivative tells us, what is the rate of change of error with respect to weight.
In neural network we use the partial derivative, since we need to find the rate of change of this error with respect to one particular weight at a time.

We use these partial derivatives in gradient descent.

**Gradient Descent** is an optimization algorithm that is used during model training to find the values of a function's parameters (coefficients) that minimize a loss function as far as possible. It is based on a convex function and tweaks its parameters iteratively to minimize a given function to its local minimum.

Partial Derivatives Calculation:

Now we start calculating the partial derivatives from the last hidden layer :
```
∂E_T/∂w5 = ∂(E1+E2)/∂w5
         = ∂E1/∂w5 + ∂E2/∂w5 
         = ∂E1/∂w5 
```
Here ∂E2/∂w5 = 0 since w5 doesn't contribute to the calculation of E2 ( as we can see in the above neural network too ), hence change in w5 will not have any effect on E2.

Next, we calculate ∂E1/∂w5 using chain rule:
```
∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5
```
The above can be interpreted as follows:
1. E1 is directly connected to a_o1, so any change in a_o1 causes a change in the value of E1.
2. a_o1 is calculated using o1, hence change in the value of o1 affects a_o1.
3. o1, is calculated using w5 and hence any change in w5 will change the value of o1 too.

In this pattern, w5 contributes to the change in E1.

Each of the above partial derivates can be calculated as follows:
```
∂E1/∂a_o1 = ∂( 1/2 (t1 - a_o1)²)/∂a_o1
          = (t1 - a_o1) * (-1) 
          = a_o1 - t1

∂a_o1/∂o1 = ∂(σ(o1))/∂o1 
          = σ(o1) * (1-σ(o1)) 
          = a_o1 * (1 - a_o1)

∂o1/∂w5 = ∂(a_h1 * w5 + a_h2 * w6)/∂w5 
        = a_h1
```
Using the above expressions, we can calculate ∂E_T/∂w5, as :
```
∂E_T/∂w5 = ∂E1/∂w5 = (a_o1 - t1) * a_o1 * (1-a_o1) * a_h1
```
Similarly,
```
∂E_T/∂w6 =  (a_o1 - t1) * a_o1 * (1-a_o1) * a_h2

∂E_T/∂w7 = (a_o2 - t2) * a_o2 * (1-a_o2) * a_h1

∂E_T/∂w8 = (a_o2 - t2) * a_o2 * (1-a_o2) * a_h2
```
At this point we have calculated all the partial derivates for the weights in the hidden layer 2.

Now, we move on to calculate the partial derivates for the weights in the hidden layer 1.

Before we calculate the partial derivatives for the weights, we can calculate some useful intermediate partial derivates which will come handy in our later calculations:
```
∂E_T/∂a_h1 = ∂(E1 + E2)/∂a_h1 = ∂E1/∂a_h1 + ∂E2/∂a_h1
  ∂E1/∂a_h1 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h1 = (a_o1 - t1) * a_o1 * (1-a_o1) * w5
  ∂E2/∂a_h1 = ∂E2/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂a_h1 = (a_o2 - t2) * a_o2 * (1-a_o2) * w7
∂E_T/∂a_h1 = (a_o1 - t1) * a_o1 * (1-a_o1) * w5 +  (a_o2 - t2) * a_o2 * (1-a_o2) * w7
```
Similarly,
```
∂E_T/∂a_h2 = (a_o1 - t1) * a_o1 * (1-a_o1) * w6 +  (a_o2 - t2) * a_o2 * (1-a_o2) * w8
```

Calculating the partial derivatives of the error w.r.t weights :
```
∂E_T/∂w1 = ∂E_T/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1

∂E_T/∂a_h1 = (a_o1 - t1) * a_o1 * (1-a_o1) * w5 +  (a_o2 - t2) * a_o2 * (1-a_o2) * w7

∂a_h1/∂h1 = ∂(σ(h1))/∂h1 = σ(h1) * (1-σ(h1)) = a_h1 * (1-a_h1)

∂h1/∂w1 = i1

∂E_T/∂w1 = (((a_o1 - t1) * a_o1 * (1-a_o1) * w5) + ((a_o2 - t2) * a_o2 * (1-a_o2) * w7)) * a_h1 * (1-a_h1) * i1
```
Similarly,
```
∂E_T/∂w2 = (((a_o1 - t1) * a_o1 * (1-a_o1) * w5) + ((a_o2 - t2) * a_o2 * (1-a_o2) * w7)) * a_h1 * (1-a_h1) * i2

∂E_T/∂w3 = (((a_o1 - t1) * a_o1 * (1-a_o1) * w6) + ((a_o2 - t2) * a_o2 * (1-a_o2) * w8)) * a_h2 * (1-a_h2) * i1

∂E_T/∂w4 = (((a_o1 - t1) * a_o1 * (1-a_o1) * w6) + ((a_o2 - t2) * a_o2 * (1-a_o2) * w8)) * a_h2 * (1-a_h2) * i2
```

### Step 4: Updating the weights
Finally, we update the weights using the gradients were just calculated
```
w1 = w1 - LR * (∂E_T/∂w1)
w2 = w2 - LR * (∂E_T/∂w2)
w3 = w3 - LR * (∂E_T/∂w3)
w4 = w4 - LR * (∂E_T/∂w4)
w5 = w5 - LR * (∂E_T/∂w5)
w6 = w6 - LR * (∂E_T/∂w6)
w7 = w7 - LR * (∂E_T/∂w7)
w8 = w8 - LR * (∂E_T/∂w8)
```

These 4 steps make one epoch. The model is trained for multiple epochs to get the best accuracy.

The below plot showing the change in Error as Learning Rate changes:
![image](https://github.com/m-shilpa/END3/blob/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/error_change_for_lr.png)

