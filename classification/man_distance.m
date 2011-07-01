function [ d ] = man_distance( v1, v2 )
%Manhattan Distance
%   Detailed explanation goes here
d = 0;
% v1 should be the longer one
if (length(v1)<length(v2))
    temp = v1;
    v1 = v2;
    v2 = temp;
end

d = d + sum(abs(v1(1:length(v2))-v2));
if (length(v1)>length(v2))
    d = d + abs(v1(length(v2)+1:length(v1)));
end

end