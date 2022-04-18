import numpy as np;
import torch;
import math;



class Rational(torch.nn.Module):
    def __init__(self,
                 Device    = torch.device('cpu')):
        # This activation function is based on the following paper:
        # Boulle, Nicolas, Yuji Nakatsukasa, and Alex Townsend. "Rational neural
        # networks." arXiv preprint arXiv:2004.01902 (2020).

        super(Rational, self).__init__();

        # Initialize numerator and denominator coefficients to the best
        # rational function approximation to ReLU. These coefficients are listed
        # in appendix A of the paper.
        self.a = torch.nn.parameter.Parameter(
                        torch.tensor((1.1915, 1.5957, 0.5, .0218),
                                     dtype  = torch.float32,
                                     device = Device));
        self.a.requires_grad_(True);

        self.b = torch.nn.parameter.Parameter(
                        torch.tensor((2.3830, 0.0, 1.0),
                                     dtype  = torch.float32,
                                     device = Device));
        self.b.requires_grad_(True);

    def forward(self, X : torch.tensor):
        """ This function applies a rational function to each element of X.

        ------------------------------------------------------------------------
        Arguments:

        X: A tensor. We apply the rational function to every element of X.

        ------------------------------------------------------------------------
        Returns:

        Let N(x) = sum_{i = 0}^{3} a_i x^i and D(x) = sum_{i = 0}^{2} b_i x^i.
        Let R = N/D (ignoring points where D(x) = 0). This function applies R
        to each element of X and returns the resulting tensor. """

        # Create aliases for self.a and self.b. This makes the code cleaner.
        a = self.a;
        b = self.b;

        # Evaluate the numerator and denominator. Because of how the * and +
        # operators work, this gets applied element-wise.
        N_X = a[0] + X*(a[1] + X*(a[2] + a[3]*X));
        D_X = b[0] + X*(b[1] + b[2]*X);

        # Return R = N_X/D_X. This is also applied element-wise.
        return torch.div(N_X, D_X);



class Sin(torch.nn.Module):
    """ This class defines a functor which implements the sin function. When
    you use the __call__ method for this class (which is defined in the Module
    superclass), it calls the forward method, which applies the sin function
    element-wise to the input. I wrote this simply to implement a sin activation
    function. """
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        return torch.sin(input);



class Neural_Network(torch.nn.Module):
    def __init__(self,
                 Num_Hidden_Layers   : int          = 3,
                 Neurons_Per_Layer   : int          = 20,   # Neurons in each Hidden Layer
                 Input_Dim           : int          = 1,    # Dimension of the input
                 Output_Dim          : int          = 1,    # Dimension of the output
                 Device              : torch.device = torch.device('cpu'),
                 Activation_Function : str          = "Tanh"):
        # For the code below to work, Num_Hidden_Layers, Neurons_Per_Layer,
        # Input_Dim, and Output_Dim must be positive integers.
        assert(Num_Hidden_Layers   > 0), "Num_Hidden_Layers must be positive. Got %du" % Num_Hidden_Layers;
        assert(Neurons_Per_Layer   > 0), "Neurons_Per_Layer must be positive. Got %u" % Neurons_Per_Layer;
        assert(Input_Dim           > 0), "Input_Dim must be positive. Got %u" % Input_Dim;
        assert(Output_Dim          > 0), "Output_Dim must be positive. Got %u" % Output_Dim;

        super(Neural_Network, self).__init__();

        # Define object attributes. Note that there is an output layer in
        # addition to the hidden layers (which is why Num_Layers is
        # Num_Hidden_Layers + 1).
        self.Input_Dim  : int = Input_Dim;
        self.Output_Dim : int = Output_Dim;
        self.Num_Layers : int = Num_Hidden_Layers + 1;


        ########################################################################
        # Set up the Layers

        # Initialize the Layers. We hold all layers in a ModuleList.
        self.Layers = torch.nn.ModuleList();

        # Append the first hidden layer. The domain of this layer is
        # R^{Input_Dim}. Thus, in_features = Input_Dim. Since this is a hidden
        # layer, its co-domain is R^{Neurons_Per_Layer}. Thus, out_features =
        # Neurons_Per_Layer.
        self.Layers.append(torch.nn.Linear(
                                in_features  = Input_Dim,
                                out_features = Neurons_Per_Layer,
                                bias         = True ).to(dtype = torch.float32, device = Device));

        # Now append the rest of the hidden layers. Each maps from
        # R^{Neurons_Per_Layer} to itself. Thus, in_features = out_features =
        # Neurons_Per_Layer. We start at i = 1 because we already created the
        # 1st hidden layer.
        for i in range(1, Num_Hidden_Layers):
            self.Layers.append(torch.nn.Linear(
                                    in_features  = Neurons_Per_Layer,
                                    out_features = Neurons_Per_Layer,
                                    bias         = True ).to(dtype = torch.float32, device = Device));

        # Now, append the Output Layer, which has Neurons_Per_Layer input
        # features, but only Output_Dim output features.
        self.Layers.append(torch.nn.Linear(
                                in_features  = Neurons_Per_Layer,
                                out_features = Output_Dim,
                                bias         = True ).to(dtype = torch.float32, device = Device));


        ########################################################################
        # Set up Activation Functions

        # Finally, set the Network's activation functions.
        self.Activation_Functions = torch.nn.ModuleList();
        if  (Activation_Function == "Tanh"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(torch.nn.Tanh());
        elif(Activation_Function == "Sin"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(Sin());
        elif(Activation_Function == "Rational"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(Rational(Device = Device));
        else:
            print("Unknown Activation Function. Got %s" % Activation_Function);
            print("Thrown by Neural_Network.__init__. Aborting.");
            exit();


        ########################################################################
        # Initialize the weight matrices, bias vectors.

        # If we're using Rational or Tanh networks, use xavier initialization.
        if(Activation_Function == "Tanh" or Activation_Function == "Rational"):
            for i in range(self.Num_Layers):
                torch.nn.init.xavier_uniform_(  self.Layers[i].weight);
                torch.nn.init.zeros_(           self.Layers[i].bias);

        # If we're using the Sin activation function, then our networks are
        # SIRENS. We use the initialization scheme proposed in the following
        # paper:
        # Sitzmann, Vincent, et al. "Implicit neural representations with
        # periodic activation functions." arXiv preprint arXiv:2006.09661 (2020).
        if(Activation_Function == "Sin"):
            # I initialize my first weight matrix to be full of zeros. This is
            # NOT what the SIREN paper says to do. However, I found that
            # following their initialization scheme causes the derivatives of my
            # network to explode. Zeroing out the first weight matrix
            # initializes U as the zero map, but eliminates blowup. 
            torch.nn.init.constant_(self.Layers[0].weight, 0);
            torch.nn.init.zeros_(   self.Layers[0].bias);

            # The SIREN paper suggests initializing the elements of every weight
            # matrix (except for the first one) by sampling a uniform
            # distribution over [-c/root(n), c/root(n)], where c > root(6),
            # and n is the number of neurons in the layer. I use c = 3 > root(6).
            #
            # Further, for simplicity, I initialize each bias vector to be zero.
            a : float = 3./math.sqrt(Neurons_Per_Layer);
            for i in range(1, self.Num_Layers):
                torch.nn.init.uniform_( self.Layers[i].weight, -a, a);
                torch.nn.init.zeros_(   self.Layers[i].bias);



    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ Forward method for the NN class. Note that the user should NOT call
        this function directly. Rather, they should call it through the __call__
        method (using the NN object like a function), which is part of the
        module class and calls forward.

        ------------------------------------------------------------------------
        Arguments:

        X: A batch of inputs. This should be a B by Input_Dim tensor, where B
        is the batch size. The ith row of X should hold the ith input.

        ------------------------------------------------------------------------
        Returns:

        If X is a B by Input_Dim tensor, then the output of this function is a
        B by Output_Dim tensor, whose ith row holds the value of the network
        applied to the ith row of X. """

        # Pass X through the hidden layers. Each has an activation function.
        for i in range(0, self.Num_Layers - 1):
            X = self.Activation_Functions[i](self.Layers[i](X));

        # Pass through the last layer (which has no activation function) and
        # return.
        return self.Layers[self.Num_Layers - 1](X);
