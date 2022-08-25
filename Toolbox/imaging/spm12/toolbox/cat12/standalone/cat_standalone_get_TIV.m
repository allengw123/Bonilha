% Batch file for getting TIV values for SPM12/CAT12 standalone installation
%
%_______________________________________________________________________
% $Id: cat_standalone_get_TIV.m 1988 2022-05-02 10:30:43Z gaser $

% data field, that will be dynamically replaced by cat_standalone.sh
matlabbatch{1}.spm.tools.cat.tools.calcvol.data_xml = '<UNDEFINED>';

% Entry for output filename
% Remove comments and edit entry if you would like to change the parameter.
% Otherwise the default value from cat_defaults.m is used.
% Or use 1st parameter field, that will be dynamically replaced by cat_standalone.sh
%matlabbatch{1}.spm.tools.cat.tools.calcvol.calcvol_name = '<UNDEFINED>';

% Entry for option to save TIV only 
% Remove comments and edit entry if you would like to change the parameter.
% Otherwise the default value from cat_defaults.m is used.
% Or use 2nd parameter field, that will be dynamically replaced by cat_standalone.sh
%matlabbatch{1}.spm.tools.cat.tools.calcvol.calcvol_TIV = '<UNDEFINED>';

% Entry to add filename to 1st column
% Remove comments and edit entry if you would like to change the parameter.
% Otherwise the default value from cat_defaults.m is used.
% Or use 3rd parameter field, that will be dynamically replaced by cat_standalone.sh
%   0 - save values only; 1 - add filename; 2 - add folder and filename 
%matlabbatch{1}.spm.tools.cat.tools.calcvol.calcvol_savenames = '<UNDEFINED>';