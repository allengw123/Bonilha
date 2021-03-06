function I=cat_vol_nanmean3(I,s,iterations)
% _________________________________________________________________________
% smooth image with nans
% ______________________________________________________________________
%
% Christian Gaser, Robert Dahnke
% Structural Brain Mapping Group (http://www.neuro.uni-jena.de)
% Departments of Neurology and Psychiatry
% Jena University Hospital
% ______________________________________________________________________
% $Id: cat_vol_nanmean3.m 1791 2021-04-06 09:15:54Z gaser $

  if ~isa('I','single'), I = single(I); end; 
  if ~exist('s','var'), s=1; end
  if ~exist('iterations','var'), iterations=1; end
  for iteration=1:iterations
    I2 = I; I3 = I;
    for i=1+s:size(I,1)-s, I2(i,:,:) = cat_stat_nanmean(I3(i-s:i+s,:,:),1); end
    for i=1+s:size(I,2)-s, I3(:,i,:) = cat_stat_nanmean(I2(:,i-s:i+s,:),2); end
    for i=1+s:size(I,3)-s, I2(:,:,i) = cat_stat_nanmean(I3(:,:,i-s:i+s),3); end  
    I(isnan(I)) = I2(isnan(I));     
  end
end 