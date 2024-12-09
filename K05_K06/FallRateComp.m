% 
% Data=dir('*dissrate*.mat');

fn='DAT_006.mat';
load([fn])

Pfs=[];
times=[];
for i=1:1:size(dataset,2)
    Pf=dataset(i).P_fast;
    time=[1:1:length(Pf)]/512;
    Pfs=cat(1,Pfs,Pf(:));
    times=cat(1,times,time(:));
    plot(time,Pf)
    hold on
end

ntime=[-0.5:1:450.5]';
Pm=depave(Pfs,times,ntime);
time2=0.5.*(ntime(1:end-1)+ntime(2:end));

plot(time2,Pm,'h','linewidth',5)
plot(time2,Pm,'r','linewidth',3)
set(gcf,'position',[259         143        1122        1202])
grid on
box off
figureset1('','Time [s]','Depth [m]','k',20)
axx=gca;
axx.XAxis.MinorTick = 'on';
axx.XAxis.MinorTickValues = axx.XAxis.Limits(1):10:axx.XAxis.Limits(2);
set(axx,'YMinorGrid','on','XMinorGrid','on','fontsize',18);
[xi,yi]=meshgrid(0:10:500,0:10:350);
plot(xi,yi,'k')
plot(xi',yi','k')
[xi2,yi2]=meshgrid(0:60:500,0:50:350);
plot(xi2,yi2,'k','linewidth',3)
plot(xi2',yi2','k','linewidth',3)


axis([0   500     0   350]);
Ax=axes('position',get(gca,'position'));
axis([0   500     0   350]);

grid on
figureset1('UVMP Table','Time [min]','Depth [m]','k',20)
set(Ax,'color','none','xaxislocation','top','yaxislocation','right',...
    'xtick',[1 2 3 4 5 6 7].*60,'xticklabel',[1 2 3 4 5 6 7],'fontsize',18,...
    'GridAlpha',1)
