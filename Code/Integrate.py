# Nonsense to add Classes directory to the Python search path.
import os;
import sys;

# Get path to Code, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Classes directory to the python path.
sys.path.append(Classes_Path);

import  torch;
from    typing          import List, Tuple;

from Network            import Network;
from Library_Term       import Library_Term;
from Weight_Function    import Weight_Function;
from Derivative         import Derivative;



def Integrate_PDE(  w               : Weight_Function,
                    U               : Network,
                    LHS_Term        : Library_Term, 
                    RHS_Terms       : List[Library_Term],
                    Mask            : torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]: 
	r"""
	We assume that the system response function, u, satisfies a PDE of the 
	form 
			f_0(u) = \sum_{k = 1}^{K} \xi_k f_k(u)
	We refer to f_0 as the "LHS Term" and {f_1, ... , f_K} as the "RHS Terms".
	Since U approximates u, we should have 
			f_0(U) \approx \sum_{i = 1}^{K} \xi_i f_k(U)
	If we multiply this by a weight function, w, we get
			f_0(U)*w \approx \sum_{i = 1}^{K} \xi_i f_k(U)*w
	This function integrates the left and right hand side of the above 
	expression over the problem domain. To understand how this works, let's
	consider a particular integral:
			\int_{\Omega} w(X) D_0 F_0(U(X)) dX
	where \Omega is the integration domain, w is a weight function, D is a
	partial derivative operator, and F(U) is a trial function. We assume both
	w and U are functions from R^n to R, for some n. We first use integration
	by parts to rewrite this integral. For this, we assume that w and all of
	its derivatives are zero on the boundary of \Omega. With this assumption,
	the integral above is equal to
			(-1)^{|D|} \int_{\Omega} D w(X) F(U(X)) dX
	where |D| is the sum of the partial derivatives in D. For example, if
	D = D_x^2 D_y^3 D_z^5, then |D| = 2 + 3 + 5 = 10. We approximate the
	integral using the composite trapezoidal rule for R^n. For this, we assume
	that the domain, \Omega, is a rectangle. That is,
		\Omega = [a_1, b_1] x ... x [a_n, b_n]
	for some a_1, ... , a_n and b_1, ... , b_n. We assume the user has
	partitioned Omega using a uniform partition (for each dimension, the
	partition points along that dimension are evenly spaced. Gird-lines along
	different dimensions, however, may have different spacing between them).
	Under this assumption, the partition partitions Omega into a set of smaller
	rectangles. Since the gird is uniform, each sub-rectangle has the same
	volume. Let V denote that volume. Further, let {X_1, ... , X_N} denote the 
	set of points in the partition that are also in the support of w. Since w 
	is, by assumption, zero along the boundary of Omega, it must be zero at 
	each point on the boundary of Omega. With this assumption, it can be shown 
	that the composite trapezoidal rule approximation to the integral is given 
	by
			(-1)^{|D|} V*( \sum_{i = 1}^{N} D w(X_i) F(U(X_i)) )
	(in the trapezoidal rule in R^n, gird points along the boundary are weighted
	differently from those inside the boundary. However, the integrand is zero
	along the boundary points, we can safely ignore those differences and
	pretend that all points are weighted evenly; the points in the summation
	above with the wrong weighting evaluate to zero anyway, so the incorrect
	weighting is moot). The weight function stores the coordinates, X_1, ... , 
	X_N as well as the volume V. 

	This function evaluates the above quantity for each library term. 
	----------------------------------------------------------------------------
	Arguments:

	w : This is a weight function. We assume that the user has already
	calculated D(w) (via the "Add_Derivative" method) on the partition.

	U : A neural network that is approximating the system response function on
	the problem domain. 

	LHS_Term: The left hand side of the PDE that the system response function 
	satisfies.

	RHS_Terms: A list of the library terms that make up the right hand side 
	of the PDE that the system response function satisfies. 

	Mask: A boolean tensor whose shape matches that of Xi. When calculating 
	the integral of the kth RHS term, we check if Mask[k] == True. If so, we 
	set the kth result to the zero tensor. Otherwise, we compute the integral 
	of the kth library term as usual.

	----------------------------------------------------------------------------
	Returns:

	A tuple. The first element holds a single element tensor whose lone entry 
	holds the value
		(-1)^{|D_0|} V*( \sum_{i = 1}^{N} D_0 w(X_i) F_0(U(X_i))) ) 
	Where f_0 = D_0 F_0. The second element is a list whose 
	kth entry holds a single element tensor whose lone value is
			(-1)^{|D_k|} V*( \sum_{i = 1}^{N} D_k w(X_i) F_k(U(X_i))) ) 
	Where f_k = D_k F_k. 
	"""
	
	############################################################################
	# First, determine the highest power of U that we need.

	Max_Pow         : int = LHS_Term.Trial_Function.Power;
	Num_RHS_Terms   : int = len(RHS_Terms);

	for i in range(Num_RHS_Terms):
		if(RHS_Terms[i].Trial_Function.Power > Max_Pow and Mask[i] == False):
			Max_Pow = RHS_Terms[i].Trial_Function.Power;


	############################################################################
	# Next, evaluate U and its powers on the coordinates stored in w.

	# First, lets fetch the coordinates that are in the support of w.
	Supported_Coords : torch.Tensor = w.Supported_Coords;

	# Next, evaluate U on these coordinates. 
	U_Coords 	: torch.Tensor  = U(Supported_Coords).view(-1);
	Num_Coords  : int           = U_Coords.numel();

	# Next, compute powers of U up to Max_Pow on the partition. We will need
	# these values for when integrating. We store them in a list.
	U_Coords_Powers = [];

	U_Coords_Powers.append(torch.ones(Num_Coords, dtype = torch.float32));
	U_Coords_Powers.append(U_Coords);

	for i in range(2, Max_Pow + 1):
		U_Coords_Powers.append(torch.pow(U_Coords, i));


	############################################################################
	# Third, compute the LHS Term integral.

	V               : float         = w.V;
	F_0_U_Coords    : torch.Tensor  = U_Coords_Powers[LHS_Term.Trial_Function.Power];
	D_0             : Derivative    = LHS_Term.Derivative;

	# Get D0(w)
	D_0_w_Coords    : torch.Tensor  = w.Derivatives[tuple(D_0.Encoding)];

	# Sum the element-wise product of F_0_U_Coords and D_0(w). This yields
	# the summation \sum_{i = 1}^{N} D_0 w(X_i) F_0(U(X_i)).
	Sum_0           : torch.Tensor  = torch.sum(torch.multiply(F_0_U_Coords, D_0_w_Coords));

	# Finally, multiply Sum by (-1)^{|D_0|}V, yielding the integral approximation
	LHS_Integral    : torch.Tensor  = torch.multiply(Sum_0, V*((-1)**D_0.Order));


	############################################################################
	# Forth, compute the RHS Term integrals.

	RHS_Integrals : List[torch.Tensor] = [];
	for k in range(Num_RHS_Terms):
		if(Mask[k] == True):
			RHS_Integrals.append(torch.sum(torch.zeros(1, dtype = torch.float32)));

		F_k_U_Coords    : torch.Tensor  = U_Coords_Powers[RHS_Terms[k].Trial_Function.Power];
		D_k             : Derivative    = RHS_Terms[k].Derivative;

		# Get D_k(w)
		D_k_w_Coords    : torch.Tensor  = w.Derivatives[tuple(D_k.Encoding)];

		# Sum the element-wise product of F_k_U_Coords and D_k(w). This yields
		# the summation \sum_{i = 1}^{N} D_k w(X_i) F_k(U(X_i)).
		Sum_k           : torch.Tensor  = torch.sum(torch.multiply(F_k_U_Coords, D_k_w_Coords));

		# Finally, multiply Sum by (-1)^{|D_k|}V, yielding the integral approximation
		RHS_Integrals.append(torch.multiply(Sum_k, V*((-1)**D_k.Order)));  


	# All done!
	return (LHS_Integral, RHS_Integrals);