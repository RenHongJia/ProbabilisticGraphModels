function [disceret_values , centroids] =Discretize(x,y)
min_sse=inf;
disceret_values=ones(length(x),1);
for k=2:6
   [index,centroids]=kmeans([x y],k, 'emptyaction','singleton') ;  
   class=round(centroids(:,2));
   h=class(index);
   sse(k-1)=sum((h-y).^2)+k;
   if(sse(k-1)<min_sse)
      min_sse=sse(k-1);
      disceret_values=index;
   end
end
