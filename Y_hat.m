function h=Y_hat(data,model,conditional_ps,p_y0,p_y1)

h=zeros(length(data),1);
fn=fieldnames(conditional_ps);
for i=1: length(data)
    pos=1;%y=0
    neg=1;%y=1
    for j=1:length(model)
        
        tmp=conditional_ps.(fn{model(j)});
        pos=pos*tmp(data(i,model(j)),1);
        neg=neg*tmp(data(i,model(j)),2);
    end   
    pos=pos*p_y0;
    neg=neg*p_y1;
    if(pos>neg)
        h(i)=0;
    else 
        h(i)=1;
    end
end