function [states] = make_simulation(i, G, perm, fluid, schedule, initState, bc, nlsolve)    
    % make rock
    kk = 10.^perm(i,:)' * milli * darcy;
    k(:,1) = kk;
    k(:,2) = kk;
    k(:,3) = 0.25 * kk;
    rock = makeRock(G, k, 0.2);
    
    % make wells
    wlocs = [32, 32, 96, 96; 
             32, 96, 96, 32];
    rinj = 0.5 * mega * 1e3 / fluid.rhoGS / year;
    wz = G.cartDims(3);

    W = [];
    for i=1:4
        W = verticalWell(W, G, rock, ...
                         wlocs(1,i), wlocs(2,i), wz-2:wz, ...
                         'InnerProduct', 'ip_tpf', ...
                         'Type'        , 'rate'  , ...
                         'Val'         , rinj    , ...
                         'comp_i'      , [0,1]   , ...
                         'refDepth'    , []      , ...
                         'name'        , ['I',int2str(i)]);
    end

    % make schedule
    schedule.control    = struct('W', W, 'bc', bc);
    schedule.control(2) = struct('W', W, 'bc', bc);
    for i=1:numel(W)
        schedule.control(2).W(i).val = 0;
    end

    model = TwoPhaseWaterGasModel(G, rock, fluid);
    [~, states] = simulateScheduleAD(initState, model, schedule, ...
                                     'NonLinearSolver', nlsolve);

end