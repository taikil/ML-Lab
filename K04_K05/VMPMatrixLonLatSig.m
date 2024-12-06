VMPlonlat

path1='./K02_K03';
path2='./K04_K05';
path3='./K05_K06';
path4='./K09_K10';
path5='./K11_K12';

hd1=dir(fullfile(path1,'DAT*dissrate*.mat'));
hd2=dir(fullfile(path2,'DAT*dissrate*.mat'));
hd3=dir(fullfile(path3,'DAT*dissrate*.mat'));
hd4=dir(fullfile(path4,'DAT*dissrate*.mat'));
hd5=dir(fullfile(path5,'DAT*dissrate*.mat'));


eps1=NaN.*zeros(200,size(hd1,1));
eps2=NaN.*zeros(200,size(hd2,1));
eps3=NaN.*zeros(200,size(hd3,1));
eps4=NaN.*zeros(200,size(hd4,1));
eps5=NaN.*zeros(200,size(hd5,1));

% K02_K03
for i=1:1:size(hd1,1)
    fn1=hd1(i).name;
    load(fullfile(path1,fn1))
    dissratei=diss.e(:,1);
    flagi=diss.flagood(:,1);
    flagi(flagi==0)=NaN;
    eps1(1:length(dissratei),i)=dissratei.*flagi;
end
