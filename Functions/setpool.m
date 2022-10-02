
function pool = setpool(input,hardreset)

% Input input
%%% 0 = turn off parpool
%%% 1 = 25% capacity
%%% 2 = 50% capacity
%%% 3 = 75% capacity
%%% 4 = 100% capacity


if ~exist('hardreset','var')
    hardreset = false;
end

core_info = evalc('feature(''numcores'')');
l_cores = regexp(core_info,'MATLAB was assigned: ','split');
l_cores = str2double(extractBefore(l_cores{2},' logical cores'));
l_cores = l_cores*(0:0.25:1);

c = parcluster;
c.NumWorkers = l_cores(input+1);

if input == 3
    disp('WARNING....')
    disp('HYPER THREADING ENABLE (75% total imaginary cores)')
    disp('MAY CAUSE OVERHEATING')
    disp('WARNING....')
elseif input == 4
    disp('WARNING....')
    disp('HYPER THREADING ENABLE (100% total imaginary cores)')
    disp('MAY CAUSE OVERHEATING')
    disp('WARNING....')
end


if ~isempty(gcp('nocreate'))
    pool = gcp('nocreate');
    if pool.NumWorkers ~= c.NumWorkers || hardreset || input == 0
        delete(pool)
        if input ~= 0
            pool = parpool(c.NumWorkers);
        end
    end

elseif input ~= 0
    pool = parpool(c.NumWorkers);
else
end
end
