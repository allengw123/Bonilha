clear all
clc

new = load('/media/bonilha/Elements/MasterSet/harvestOutput/Patients_rerun/MUSPR0033/pre/T1_MUSPR0033_pre_lime.mat');
old = load('/media/bonilha/Elements/MasterSet/harvestOutput/Patients_old/MUSPR0033/pre/T1_MUSPR0033_pre_lime.mat');


fn = fieldnames(new);

fails = [];
for f = 1:numel(fn)
    new_data = new.(fn{f});
    old_data = old.(fn{f});

    if isstruct(old_data)
        fn_new = fieldnames(new_data);
        fn_old = fieldnames(old_data);

        if all(strcmp(fn_new,fn_old))
            disp(['PASS - Fieldnames for ',fn{f},' match'])
        else
            err_msg = ['FAIL - Fieldnames for ',fn{f},' do not match'];
            disp(err_msg)
            fails = [fails; {err_msg}];
            continue
        end
        
        fails = checkstruct(new_data,old_data,fails,fn{f});
    elseif isnumeric(new_data)
        err_margin = (sum(new_data,'all') - sum(new_data,'all'))/sum(old_data,'all')*100;
        if err_margin < 1
            disp(['PASS - Field values for',fn{f},' within % error margin of ',num2str(err_margin)])
        else
            err_msg = ['FAIL - Field values for ',fn{f},' not within % error margin with ',num2str(err_margin)];
            disp(err_msg)
            fails = [fails; {err_msg}];
        end
    else
        error()
    end
end

%% Function
function fails = checkstruct(new_data,old_data,fails,field)

fn_new = fieldnames(new_data);
fn_old = fieldnames(old_data);

for i = 1:numel(fn_new)
    dat_fn_new = new_data.(fn_new{i});
    dat_fn_old = old_data.(fn_old{i});

    if ischar(dat_fn_new)
        if all(dat_fn_new == dat_fn_old,'all')
            disp(['PASS - Field characters ',fn_new{i},' for ',field,' match'])
        else
            err_msg = ['FAIL - Field characters ',fn_new{i},' for ',field,' did not match'];
            disp(err_msg)
            fails = [fails; {err_msg}];
            continue
        end
    elseif isnumeric(dat_fn_new)
        dim = ndims(dat_fn_new);
        myMatrices = cat( dim+1, dat_fn_new, dat_fn_old);
        nanLocations = isnan( myMatrices );
        allNaNs = sum( nanLocations, dim+1 ) == 2;

        dat_fn_new(allNaNs) = 0;
        dat_fn_old(allNaNs) = 0;
        
        nan_new = sum(isnan(dat_fn_new),'all');
        nan_old = sum(isnan(dat_fn_old),'all');

        err_margin = (sum(dat_fn_new,'all','omitnan') - sum(dat_fn_old,'all','omitnan'))/sum(dat_fn_old,'all','omitnan')*100;
        if isnan(err_margin)
            err_margin = dat_fn_new-dat_fn_old;
        end
        if err_margin < 1
            disp(['PASS - Field values ',fn_new{i},' for ',field,' within % error margin of ',num2str(err_margin)])
            if (nan_new+nan_old)>0
                disp(['......with ',num2str((nan_new+nan_old)/numel(dat_fn_new)*100),' percent of values mismatched as NaN ',num2str((nan_new+nan_old))])
            end
        else
            err_msg = ['FAIL - Field values ',fn_new{i},' for ',field,' not within % error margin with ',num2str(err_margin)];
            disp(err_msg)
            fails = [fails; {err_msg}];
        end
    elseif isa(dat_fn_new,'struct')
        fails = checkstruct(dat_fn_new,dat_fn_old,fails,field);
    else
        error()
    end
end
end