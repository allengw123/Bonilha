struc_mat = [[0,1,3,0];[1,0,0,0];[3,0,0,0];[0,0,0,0]]; 
func_mat = [[0,4,3,1];[4,0,2,2];[3,2,0,1];[1,2,1,0]]; 


figure;
imagesc(struc_mat);
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gcf,'color',[1 1 1]);
cb=colorbar;
set(cb,'Ticks',[0 10],'FontSize',16,'Location','southoutside','Limits',[0 10])
ylabel(cb,'	','FontSize',16)
axis('square')
ccm=customcolormap([0 1],{'#E28E40','#6E3908'});
colormap(ccm)

figure;
imagesc(func_mat);
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
set(gcf,'color',[1 1 1]);
cb=colorbar;
set(cb,'Ticks',[0 10],'FontSize',16,'Location','southoutside','Limits',[0 10])
ylabel(cb,'	','FontSize',16)
axis('square')
ccm=customcolormap([0 1],{'#4094E2','#063058'});
colormap(ccm)