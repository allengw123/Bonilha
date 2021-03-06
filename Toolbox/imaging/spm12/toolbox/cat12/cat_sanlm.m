function cat_sanlm(in, v, f, rician)
% FORMAT cat_sanlm(in, v, f, rician)
% 
% Spatial Adaptive Non Local Means Denoising Filter
%
% v - size of search volume (M in paper)
% f - size of neighborhood (d in paper)
% rician - use rician noise distribution
%
% *                          Details on SANLM filter                        
% ***************************************************************************
% *  The SANLM filter is described in:                                      *
% *                                                                         *
% *  Jose V. Manj—n, Pierrick Coupe, Luis Mart’-bonmat’, Montserrat Robles  *
% *  and D. Louis Collins.                                                  *
% *  Adaptive Non-Local Means Denoising of MR Images with Spatially Varying *
% *  Noise Levels. Journal of Magnetic Resonance Imaging, 31,192-203, 2010. *                                                       
% *                                                                         *
% ***************************************************************************/
%
% ______________________________________________________________________
%
% Christian Gaser, Robert Dahnke
% Structural Brain Mapping Group (http://www.neuro.uni-jena.de)
% Departments of Neurology and Psychiatry
% Jena University Hospital
% ______________________________________________________________________
% $Id: cat_sanlm.m 1791 2021-04-06 09:15:54Z gaser $

rev = '$Rev: 1791 $';

disp('Compiling cat_sanlm.c')

pth = fileparts(which(mfilename));
p_path = pwd;
cd(pth);
mex -O cat_sanlm.c sanlm_float.c 
cd(p_path);

cat_sanlm(in, v, f, rician);

return
