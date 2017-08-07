function rcv=LOOCV(x,y,h)
rcv=0;
u=x* inv((x.') * x) * (x.');
for i=1:length(x)
    rcv=rcv+( ((y(i) -h(i)) / (1-u(i,i)) )^2);
end