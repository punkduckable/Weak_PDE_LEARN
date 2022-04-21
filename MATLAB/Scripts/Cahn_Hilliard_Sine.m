% Cahn-Hilliard equation: D_t U = -5(D_{x}^2 U) - (0.5)D_{x}^4 U) + 
% 5*(D_{x}^2 (U^3))
% Set up the problem domain. I want this to run on
%       (t, x) in [0, 20] x [-10, 10]
x_l     = -10;
x_h     = 10;
t_l     = 0;
t_h     = 10;
Nt      = 201;
Domain  = [x_l, x_h];
Tspan   = linspace(t_l, t_h, Nt);

% Set up the spinop for the Cahn-Hilliard equation. For this problem:
%       L(U)    = -5 (D_x^2 U) - 0.5 D_{x}^4 U 
%       N(U)    = 5*D_x^2 (U^3).
%       Init(x) = (x^3/1000)*sin(.08*pi*x^2)
S           = spinop(Domain, Tspan);
S.lin       = @(u) -5*diff(u, 2) - .5*diff(u, 4);
S.nonlin    = @(u) 5*diff(u.^3, 2);
S.init      = chebfun(@(x) (x^3/1000)*sin(.08*pi*x^2), Domain, 'vectorize');

% Solve!
disp("Solving...");
Nx      = 256;
Dt      = .0002;
U_Cheb  = spin(S, Nx, Dt, 'plot', 'off');


% Make the dataset....
disp("Writing solution to array...");
usol    = zeros(Nx, Nt);
x_vals  = linspace(x_l, x_h, Nx + 1);
x_range = x_vals(1:(end - 1));
for t = 1:Nt
    usol(:, t) = U_Cheb{t}(x_range);
end

% Save!
disp("Saving...");
t = Tspan;
x = x_range;
save('../Data/Cahn_Hilliard_Sine.mat','t','x','usol');

% Plot!
figure(1);
hold on;
set(gca, 'FontSize', 12);

pcolor(t, x, usol); shading interp, colorbar, axis tight, colormap(jet);

xlabel('time (s)');
ylabel('position (m)');
title("Cahn-Hilliard equation dataset (Sine IC)");