clear;
clc;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%Load Data%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
aa=load('processed.cleveland.data','r');
b=load('processed.hungarian.data','r');
c=load('processed.switzerland.data','r');
d=load('processed.va.data','r');
data=[aa;b;c;d];
discrete=[2,3,6,7,9,11,12,13];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%PrePorcessing%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%Make problem binary%%%%%%%%%%%%%%%%
for j=1:length(data)
   if(data(j,14)~=0)
       data(j,14)=1;
   end
end

%%Replacing missing values with mean and mode%%
% for i=1:14
%     a=~isnan(data(:,i));
%     if(any(discrete==i))
%         mean_data(i)=mode(data(find(a),i));   
%     else
%         mean_data(i)=mean(data(find(a),i));    
%     end
%     for j=1:length(data)
%         if(isnan(data(j,i)))
%             data(j,i)=mean_data(i);
%         end
%     end
%     
% end
data=Imputation(data,NaN);

%%%%%%%%%%%%Shuffling data%%%%%%%%%%%%%%%%
data=datasample(data,length(data),'Replace',false);
N=size(data,2)-1;

%%%Part1- alef: plot feature vs Y%%%
for i=1:6
subplot(3,2,i)
scatter(data(:,i),data(:,N+1));
hold on 
xlabel(strcat('Feature ',num2str(i)));
ylabel('Y');
end
figure;
for i=7:13
subplot(4,2,i-6)
scatter(data(:,i),data(:,N+1));
xlabel(strcat('Feature ',num2str(i)));
ylabel('Y');
end

%%%%%%%%%%%%Discretize%%%%%%%%%%%%%%%%
cont=[1,4,5,8,10];
[data(:,1) ,~]=Discretize(data(:,1) ,data(:,14));
[data(:,4) ,~]=Discretize(data(:,4) ,data(:,14));
[data(:,5) ,~]=Discretize(data(:,5) ,data(:,14));
[data(:,8) ,~]=Discretize(data(:,8) ,data(:,14));
[data(:,10),~]=Discretize(data(:,10),data(:,14));
C=zeros(13,1);
for i=1:13
    C(i)=length(unique(data(:,i)));
end



%%Shift containing zero attribute 1 unit and scaling%%%
for i=1:12
    data_max=max(data(:,i));
    data_min=min(data(:,i))-1;
    data(:,i)=(data(:,i)-data_min);
end

for i=1:length(data)
    data(i,13)=(data(i,13)-4);
    if(data(i,13)<0)
        data(i,13)=1;
    end
end

ts_data=data(1:200,:);
tr_data=data(201:920,:);
% val_data=data(801:920,:);
m=length(tr_data);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%Naive Bayes Classifier%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%Compute conditional probabilities for each attribute%%%
p_y0=sum(tr_data(:,14)==0)/m;
p_y1=sum(tr_data(:,14)==1)/m;
conditional_ps=struct('P1',0,'P2',0,'P3',0,'P4',0,'P5',0,'P6',0,'P7',0,'P8',0,'P9',0,'P10',0,'P11',0,'P12',0,'P13',0);
fn=fieldnames(conditional_ps);
for i=1:N
    p=Probs(tr_data,i,C(i));
    conditional_ps.(fn{i})=p;
end

%%%Test data prediction%%%%
model=(1:N);
h=Y_hat(ts_data,model,conditional_ps,p_y0,p_y1); 
all_err=sum(abs(h-ts_data(:,14)))/length(h);

%%LOOCV error

%%%Train data prediction%%%%
hh=Y_hat(tr_data,model,conditional_ps,p_y0,p_y1); 
xx=tr_data(:,1:N);
yy=tr_data(:,14);
rcv_all=LOOCV(xx,yy,hh);
% % mcr=crossval('mcr',[ones(1,100);(1:length(ts_data))].',yy,'Predfun',func);

%%%%%%%%%%%%%%%%%%%%%%%
%%%feature selection%%%
%%%%%%%%%%%%%%%%%%%%%%%

%%%1-forward greedy search
rcv_current=inf;
rcv_selected=inf;
model=2;
while(rcv_selected <= rcv_current)
    k=1;
    while(k<N+1)
        if(any(model==k))
            rcv_new(k)=inf;
            k=k+1;
            continue;
        end
        model_cr=[model k];        
        h=Y_hat(tr_data,model_cr,conditional_ps,p_y0,p_y1); 
        xx=tr_data(:,model_cr);
        yy=tr_data(:,14);
        rcv_new(k)=LOOCV(xx,yy,h);
        k=k+1;
        rcv_current=rcv_selected;
    end
    [rcv_selected,selected_index]=min(rcv_new);
    model=[model selected_index];
end
best_forward_model=model(1:length(model)-1);
    


%%%2-backward greedy search
selected_rcv(1)=rcv_all;
omitted=0;
while(length(omitted)<N)    
    j=N;
    rcv=zeros(j,1);
    while(j>0)
        if(any(omitted==j))
           rcv(j)=inf;
           j=j-1;
           continue;
        end
        model=Remove((1:13),[j omitted]);
        X=tr_data(:,model);
        Y=tr_data(:,14);
        h=Y_hat(tr_data,model,conditional_ps,p_y0,p_y1); 
        rcv(j)=LOOCV(X,Y,h);
        j=j-1;        
    end
    [rcv_omitted,omitted_index]=min(rcv);
    omitted=[omitted omitted_index];
    selected_rcv(length(omitted))=rcv_omitted; 
       
end
[mm ,in]=min(selected_rcv);
best_backward_model=Remove((0:N),omitted(1:in+1));


%%%Recreating seleted model

h1=Y_hat(ts_data,best_forward_model,conditional_ps,p_y0,p_y1);
fw_err=sum(abs(h1-ts_data(:,14)))/length(h1);
h1_tr=Y_hat(tr_data,best_forward_model,conditional_ps,p_y0,p_y1); 
rcv_fw_model=LOOCV(tr_data(:,best_forward_model),tr_data(:,14),h1_tr);

h2=Y_hat(ts_data,best_backward_model,conditional_ps,p_y0,p_y1);
bw_err=sum(abs(h2-ts_data(:,14)))/length(h2);
h2_tr=Y_hat(tr_data,best_backward_model,conditional_ps,p_y0,p_y1); 
rcv_bw_model=LOOCV(tr_data(:,best_backward_model),tr_data(:,14),h2_tr);


clc;
fprintf('\nBest forward model includes: ')
for p=1:length(best_forward_model)
    fprintf('X_%d | ',best_forward_model(p));
end

fprintf('\nBest backward model includes: ')
for p=1:length(best_backward_model)
    fprintf('X_%d | ',best_backward_model(p));
end
fprintf('\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%Knowledge Expert DAG%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%create DAG structure
N = 14; 
dag = zeros(N,N);
Y =14; X=[4,5,6,7,8,9,11,12,13]; 
dag(Y,X) = 1;
dag(9,3)=1;
Y10=[7,9,11];
dag(Y10,10)=1;
dag(12,1)=1;
Y2=[3,4,5,6,8,12];
dag(Y2,2)=1;


node_size = ones(1,N);
Y=14; X=(1:13);
% dag(Y,X) = 1; naive bayes

node_size(Y)=2;
node_size(X)=C;

naive_net=mk_bnet(dag,node_size,'discrete',Y,'observed',X);
naive_net.CPD{Y}=tabular_CPD(naive_net,Y);
for i=1:13
   naive_net.CPD{X(i)}=tabular_CPD(naive_net,X(i)); 
end




t=[tr_data(:,X) , tr_data(:,Y)+1];

t=num2cell(t');
naive_net=learn_params(naive_net,t);   

engine = jtree_inf_engine(naive_net);
evidence = cell(1,N);

% compute probabilities
test_T=num2cell( ts_data.');
for i=1:length(ts_data)
 evidence(1:13) = test_T(1:13,i);
 [engin2,ll]=enter_evidence(engine, evidence);
  marginal(i)= marginal_nodes(engin2, Y);
   if(marginal(i).T(1)>=marginal(i).T(2))
       prediction(i)=0;
   else
       prediction(i)=1;
   end
   
end

bnt_err1=sum(abs(prediction.'-ts_data(:,14)))/length(ts_data);

%%%%LOOCV Calculation%%%

tr_T=num2cell( tr_data.');
for i=1:length(tr_T)
 evidence(1:13) = tr_T(1:13,i);
 [engin2,ll]=enter_evidence(engine, evidence);
  marginal(i)= marginal_nodes(engin2, Y);
   if(marginal(i).T(1)>=marginal(i).T(2))
       h_T(i)=0;
   else
       h_T(i)=1;
   end
   
end
rcv_bnt1=LOOCV(tr_data(:,1:13),tr_data(:,14),h_T);
   

%%%%Second graph%%%%%%

%create DAG structure
N = 14; 
dag = zeros(N,N);
Y =14; X=[3,5,2]; 
dag(Y,X) = 1;
dag(5,2)=1;
dag(5,1)=1;
dag(3,2)=1;



node_size = ones(1,N);
Y=14; X=(1:13);
% dag(Y,X) = 1; naive bayes

node_size(Y)=2;
node_size(X)=C;

naive_net=mk_bnet(dag,node_size,'discrete',Y,'observed',X);
naive_net.CPD{Y}=tabular_CPD(naive_net,Y);
for i=1:13
   naive_net.CPD{X(i)}=tabular_CPD(naive_net,X(i)); 
end




t=[tr_data(:,X) , tr_data(:,Y)+1];

t=num2cell(t');
naive_net=learn_params(naive_net,t);   

engine = jtree_inf_engine(naive_net);
evidence = cell(1,N);

% compute probabilities
test_T=num2cell( ts_data.');
for i=1:length(ts_data)
 evidence(1:13) = test_T(1:13,i);
 [engin2,ll]=enter_evidence(engine, evidence);
  marginal(i)= marginal_nodes(engin2, Y);
   if(marginal(i).T(1)>=marginal(i).T(2))
       prediction(i)=0;
   else
       prediction(i)=1;
   end
   
end

bnt_err2=sum(abs(prediction.'-ts_data(:,14)))/length(ts_data);

%%%%LOOCV Calculation%%%

tr_T=num2cell( tr_data.');
for i=1:length(tr_T)
 evidence(1:13) = tr_T(1:13,i);
 [engin2,ll]=enter_evidence(engine, evidence);
  marginal(i)= marginal_nodes(engin2, Y);
   if(marginal(i).T(1)>=marginal(i).T(2))
       h_T(i)=0;
   else
       h_T(i)=1;
   end
   
end
rcv_bnt2=LOOCV(tr_data(:,1:13),tr_data(:,14),h_T);
   




figure;
subplot(2,1,1);
funcs={'NB';'FWS';'BWS';'DAG1';'DAG2'};
bar([all_err,fw_err,bw_err,bnt_err1,bnt_err2]);
set(gca,'xticklabel',funcs)
ylabel('Error');
title('Test Error Rate');


subplot(2,1,2);
bar([rcv_all,rcv_fw_model,rcv_bw_model,rcv_bnt1,rcv_bnt2]);
set(gca,'xticklabel',funcs)
ylabel('Rcv');
title('LOOCV');
