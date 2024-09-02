function [schedule] = gen_schedule(W, bc, timesteps)

    schedule.control = struct('W',W, 'bc',bc);
    schedule.step.val = timesteps;
    schedule.step.control = ones(numel(timesteps),1);

end