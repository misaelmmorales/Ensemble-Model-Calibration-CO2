%% Main variables
addpath 'D:\MATLAB\mrst-2024a'
set(0,'DefaultFigureWindowStyle','docked')
% Import MRST module
mrstModule add co2lab mimetic matlab_bgl
mrstModule add ad-core ad-props ad-blackoil mrst-gui
clear; clc

%% Grid
nx = 64; ny = 64; nz = 8;
dx = 50; dy = 50; dz = 25;
G = cartGrid([nx,ny,nz], [nx*dy, ny*dy, nz*dz]*meter);
G.nodes.coords(:,3) = G.nodes.coords(:,3)+1000;
G = computeGeometry(G);

%% Grid 2
m = nx/2;
[x,y] = meshgrid(0:2*m);

T = 2 - 3*membrane(1,m) - 0.2*sin(pi*x/12)*sin(pi*y/7)  - 0.2*sin(pi*(x+y)/5)   - 0.10*sin(pi*(x.*y)/100) - 0.1*rand(size(x));
B = 2 - 3*membrane(1,m) - 0.2*sin(pi*x/15).*sin(pi*y/8) - 0.1*sin(pi*(x+y+5)/5) - 0.25*sin(pi*(x.*y)/80)  + 18.0;

G = tensorGrid(0:2*m, 0:2*m, 0:nz);
num = prod(G.cartDims(1:2)+1);
for k=1:n+1
   G.nodes.coords((1:num)+(k-1)*num,3) = T(:) + (k-1)/n*(B(:)-T(:));
end
clear x y T B;

G.nodes.coords(:,1:2) = G.nodes.coords(:,1:2)*50;
G.nodes.coords(:,3) = G.nodes.coords(:,3)*32 + 1000;
G = computeGeometry(G);

figure(1); clf; plotCellData(G, G.cells.centroids(:,3)); view(-45,70); colormap jet

perm = load('data_1272_64x64x8.mat').perm;
perm = reshape(perm, [1272, nx*ny*nz]);

%% Initial State
gravity on;  g = gravity;
rhow = 1000;
%P0 = rhow * G.cells.centroids(1,3) * g(3);
P0 = 3000 * psia;
initState.pressure = repmat(P0, G.cells.num, 1);
initState.s = repmat([1, 0], G.cells.num, 1);
initState.sGmax = initState.s(:,2);

%% Fluid
co2     = CO2props();             % load sampled tables of co2 fluid properties
p_ref   = 30 * mega * Pascal;     % choose reference pressure
t_ref   = 94 + 273.15;            % choose reference temperature, in Kelvin
rhoc    = co2.rho(p_ref, t_ref);  % co2 density at ref. press/temp
cf_co2  = co2.rhoDP(p_ref, t_ref) / rhoc; % co2 compressibility
cf_wat  = 0;                      % brine compressibility (zero)
cf_rock = 4.35e-5 / barsa;        % rock compressibility
muw     = 8e-4 * Pascal * second; % brine viscosity
muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity

% Use function 'initSimpleADIFluid' to make a simple fluid object
fluid = initSimpleADIFluid('phases', 'WG'           , ...
                           'mu'  , [muw, muco2]     , ...
                           'rho' , [rhow, rhoc]     , ...
                           'pRef', p_ref            , ...
                           'c'   , [cf_wat, cf_co2] , ...
                           'cR'  , cf_rock          , ...
                           'n'   , [2 2]);

% Modify relative permeability curves and capillary pressure
srw = 0.27;
src = 0.20;
fluid.krW = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
fluid.krG = @(s) fluid.krG(max((s-src)./(1-src), 0));
pe = 5 * kilo * Pascal;
pcWG = @(sw) pe * sw.^(-1/2);
fluid.pcWG = @(sg) pcWG(max((1-sg-srw)./(1-srw), 1e-5));

%% Boundary Conditions
bc = [];
vface_ind = (G.faces.normals(:,3) == 0);
bface_ind = (prod(G.faces.neighbors, 2) == 0);
bc_face_ix = find(vface_ind & bface_ind);
bc_cell_ix = sum(G.faces.neighbors(bc_face_ix, :), 2);
p_face_pressure = initState.pressure(bc_cell_ix);
bc = addBC(bc, bc_face_ix, 'pressure', p_face_pressure, 'sat', [1,0]);

%% Nonlinear solver
lsolve  = BackslashSolverAD('maxIterations', 50, 'tolerance', 1e-2);
nlsolve = NonLinearSolver('useRelaxation'  , true, ...
                          'maxTimestepCuts', 5   , ...
                          'maxIterations'  , 8  , ...
                          'useLinesearch'  , true, ...
                          'LinearSolver'   , lsolve);

%% Schedule
Tinj  = 5*year;
dTinj = year/12;
nTinj = Tinj / dTinj;

Tmon  = 500*year;
dTmon = 10*year;
nTmon = Tmon / dTmon;

tsteps = [Tinj, dTinj, nTinj; 
          Tmon, dTmon, nTmon];

dT = rampupTimesteps(Tinj, dTinj, 6);

schedule.step.val     = [dT                ; repmat(dTmon, nTmon, 1)];
schedule.step.control = [ones(numel(dT), 1); ones(nTmon, 1) * 2];

%% Temporary single simulation

i = 666;

[states,W,rock] = make_simulation(i, G, perm, fluid, schedule, initState, bc, nlsolve);

figure(2); clf; plotCellData(G, log10(convertTo(rock.perm(:,1), milli*darcy))); 
plotWell(G,W,'color','k'); colormap jet; colorbar; view(-45,70)

figure(3); clf; plotToolbar(G, states); plotWell(G,W); colormap jet; colorbar; view(-45,70)

save('states.mat', 'states')

%% Run parallel simulations
parfor i=1:4
    [states,W,rock] = make_simulation(i, G, perm, fluid, schedule, initState, bc, nlsolve);
    parsave(sprintf('states/states_%d', i-1), states);
    fprintf('Simulation %i done\n', i-1)
end
disp('... All Done!');

%% Visualize
%{
figure(1); clf; plotCellData(G, G.cells.centroids(:,3)); 
colormap jet; view(-40,75); colorbar

figure(2); clf;
plotCellData(G, log10(convertTo(rock.perm(:,1),milli*darcy))); 
plotWell(G,W,'color','k'); view(-40,75); colormap jet; colorbar

figure(3); clf; 
plotToolbar(G, states); 
plotWell(G,W,'color','k'); 
view(-40,75); colormap jet; colorbar
%}