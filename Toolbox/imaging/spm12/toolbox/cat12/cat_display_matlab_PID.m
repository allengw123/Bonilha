function varagout = cat_display_matlab_PID
% cat_display_matlab_PID (in development!)
% ______________________________________________________________________
%
% Display exor return PID of this (Linux/Mac) or the last started 
% (Windows) MATLAB instace.
% ______________________________________________________________________
%
% Christian Gaser, Robert Dahnke
% Structural Brain Mapping Group (http://www.neuro.uni-jena.de)
% Departments of Neurology and Psychiatry
% Jena University Hospital
% ______________________________________________________________________
% $Id: cat_display_matlab_PID.m 1791 2021-04-06 09:15:54Z gaser $

  % get PID
  pid = feature('getpid'); 
  
  % display PID
  if nargout==0
    if isnumeric(pid) && ~isempty(pid)
      fprintf('CAT parallel processing with MATLAB PID: %d\n',pid);
    else
      fprintf('CAT parallel processing with MATLAB PID: unknown %s PID %d\n',computer,pids);
    end
  else 
    varagout{1} = pid; 
  end
  
end
