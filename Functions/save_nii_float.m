function save_nii_allen(vol,save_name)
vol.hdr.dime.datatype = 16;
vol.hdr.dime.bitpix = 16;

vol.img(isnan(vol.img)) = 0;

save_nii(vol,save_name)
end