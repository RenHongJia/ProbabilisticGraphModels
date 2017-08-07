function index=MostSimilare(R,data,missed_value)

attrs=find(R~=0);
standard_dev=sqrt(var(data));
IMMV=zeros(length(data),1);
for i=1:length(data)
    for k=1:length(attrs)
        sim=0.5* (1+ exp(-(((R(attrs(k))-data(i,attrs(k)))/standard_dev(k))^2)));
        IMMV(i)=  IMMV(i)+sim ;
    end
end

[~,index]=max(IMMV);