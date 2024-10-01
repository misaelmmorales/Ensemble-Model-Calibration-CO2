function [states, W, rock] = make_simulation(i, G, perm, fluid, schedule, initState, bc, nlsolve)    
    % make rock
    kk = 10.^perm(i,:)' * milli * darcy;
    kp(:,1) = kk;
    kp(:,2) = kk;
    kp(:,3) = 0.25 * kk;
    rock = makeRock(G, kp, 0.2);
   
    % make wells
    w1 = G.cartDims(1) * 0.25;
    w2 = G.cartDims(1) * 0.75;
    wz = G.cartDims(3);

    wlocs = [w1, w1, w2, w2; 
             w1, w2, w2, w1];

    rinj = 0.5 * mega * 1e3 / fluid.rhoGS / year;

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
    %[~, states] = simulateScheduleAD(initState, model, schedule, 'NonLinearSolver', nlsolve);
    [~, states] = simulateScheduleAD(initState, model, schedule);

end

