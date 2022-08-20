input_error_log = '~/Downloads/boxsync_error_log.txt';

% Read error log
fid=fopen(input_error_log);
problem_sbjs = cell(0,1);
while true
    tline = fgetl(fid);
    if ~ischar(tline)
        break
    end
    file = extractBetween(tline,'ERROR : ',':');
    problem_sbjs{end+1,1} = file;
end
fclose(fid);

% Find unique subjects
problem_sbjs = cellfun(@fileparts,problem_sbjs,'UniformOutput',false);
problem_sbjs = unique(problem_sbjs(cellfun(@ischar,problem_sbjs)));