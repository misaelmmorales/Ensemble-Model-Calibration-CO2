function [outputArg1,outputArg2] = make_simulationVE(i, Gt, perm)
    
    % make rock
    kk = 10.^perm(i,:)' * milli * darcy;
    kp(:,1) = kk;
    kp(:,2) = kk;
    kp(:,3) = 0.25 * kk;
    rock = makeRock(G, kp, 0.2);
    rock2d = averageRock(rock, Gt);

    % make wells
    w1 = Gt.cartDims(1) * 0.25;
    w2 = Gt.cartDims(1) * 0.75;
    wlocs = [w1, w1, w2, w2; 
             w1, w2, w2, w1];
    rinj = 0.5 * mega * 1e3 / fluid.rhoGS / year;
    W2d = [];
    for i=1:4
        W2d = verticalWell(W2d, Gt, rock2d, ...
                            wlocs(1,i), wlocs(2,i), 1, ...
                            'InnerProduct', 'ip_tpf', ...
                            'Type'        , 'rate'  , ...
                            'Val'         , rinj    , ...
                            'comp_i'      , [1,0]   , ...
                            'refDepth'    , []      , ...
                            'name'        , ['I',int2str(i)]);
    end
    


end