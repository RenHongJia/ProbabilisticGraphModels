function s=Remove(a,b)
Acommon = intersect(a,b);
s = setxor(a,Acommon);