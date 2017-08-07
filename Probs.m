function P=Probs(input,x_index,C)
y=input(:,14);
% m=length(data);
% C=length(unique(data(:,x_index)));
P=zeros(C,2);

for i=1:C
    
    P(i,1)=sum(y==0 & input(:,x_index)==i )/(sum(y==0));
    P(i,2)=sum(y==1 & input(:,x_index)==i )/(sum(y==1));
end
l=2;
if(any(P(:,1)==0))
    for i=1:C
        P(i,1)=(sum(y==0 & input(:,x_index)==i)+l)/((sum(y==0))+(l*C));
    end
end
if(any(P(:,2)==0))
    for i=1:C
        P(i,2)=(sum(y==1 & input(:,x_index)==i )+l)/((sum(y==0))+(l*C));
    end
end

