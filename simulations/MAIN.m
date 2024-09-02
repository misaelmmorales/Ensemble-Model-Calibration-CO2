%% Main variables
set(0,'DefaultFigureWindowStyle','docked')
% Import MRST module
mrstModule add co2lab mimetic matlab_bgl
mrstModule add ad-core ad-props ad-blackoil mrst-gui
clear;clc

% Define global variables
dims = 64;

%% Make Grid
[G, Gt, ~, ~, bcIx, bcIxVE] = makeModel(32,1);
save('grids/G.mat', 'G')
save('grids/Gt.mat', 'Gt')

figure(1); clf; plotCellData(G, G.cells.centroids(:,3)); view(3); colormap jet; colorbar

%% Make Initial State
gravity on;  g = gravity;
rhow = 1000;
P0 = 4000*psia;
initState.pressure = repmat(P0, dims*dims, 1);
initState.s = repmat([1, 0], G.cells.num, 1);
initState.sGmax = initState.s(:,2);

%% Make Fluid
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

%% Make Boundary Conditions
bc = [];
vface_ind = (G.faces.normals(:,3) == 0);
bface_ind = (prod(G.faces.neighbors, 2) == 0);
bc_face_ix = find(vface_ind & bface_ind);
bc_cell_ix = sum(G.faces.neighbors(bc_face_ix, :), 2);
p_face_pressure = initState.pressure(bc_cell_ix);
bc = addBC(bc, bc_face_ix, 'pressure', p_face_pressure, 'sat', [1,0]);

%% Generate Models
perm = load('data/perm_64x64.mat').perm';
poro = load('data/poro_64x64.mat').poro';
facies = load('data/facies_64x64.mat').facies';

R_inj = 2 * mega * 1e3 / fluid.rhoGS / year;
timesteps = rampupTimesteps(10*year, year/2, 8);

conversion = rhoc * year / 1e3 / mega;

%% Run Simulation
parfor i=0:999
    [rock]            = gen_rock(i, perm, poro, facies);
    [W, ~]            = gen_wells(G, rock, R_inj);
    [schedule]        = gen_schedule(W, bc, timesteps);
    [wellSol, states] = gen_ADsimulation(G, rock, fluid, initState, schedule);
    parsave(sprintf('states/states_%d', i), states);
    parsave(sprintf('rock/rock_%d', i), rock);
    fprintf('Simulation %i done\n', i)
end
disp('... All Done!');

%% END