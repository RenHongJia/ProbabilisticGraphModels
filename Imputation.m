function data_n=Imputation(data,missed_value)
% close all;
% clear;
% clc;
% data=load('Dataset2.csv');
% missed_value=0;

complete_data=zeros(1,size(data,2));
incomplete_data=zeros(1,size(data,2));
ic=0;
c=0;
for i=1:length(data)
    
    if(~any(isnan(data(i,:)))==1)
        complete_data=[complete_data ; data(i,:)];
        c=[c, i];
    else
        incomplete_data=[incomplete_data ; data(i,:)];
        ic=[ic , i];
    end
end

complete_data  =  complete_data(2:length(  complete_data),:);
incomplete_data=incomplete_data(2:length(incomplete_data),:);


for i=1:length(incomplete_data)
    index=MostSimilare(incomplete_data(i,:),complete_data,0);
    missed_attr=find(isnan(incomplete_data(i,:)));
    incomplete_data(i,missed_attr)=complete_data(index,missed_attr);
end
c =c (2:length(c));
ic=ic(2:length(ic));
data_n(ic,:)=incomplete_data;
data_n(c,:)=complete_data;