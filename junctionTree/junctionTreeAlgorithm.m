clc;
n = 5;

psis = cell(n-1, 1);
phis = cell(n-2, 1);
phisStar = cell(n-2, 1);
phisDoubleStar = cell(n-2, 1);

for i = 1:(n-2)
    phis{i} = ones(1,2);
    phisStar{i} = ones(1,2);
    phisDoubleStar{i} = ones(2,1);
end

psis{1} = [ 0.1 0.7 ; 0.8 0.3 ];
psis{2} = [ 0.5 0.1 ; 0.1 0.5 ];
psis{3} = [ 0.1 0.5 ; 0.5 0.1 ];
psis{4} = [ 0.9 0.3 ; 0.1 0.3 ];

%Forward Pass (Collect phase)
for i = 1:(n-2)
    phisStar{i} = [ (psis{i}(1,1)+psis{i}(2,1)), (psis{i}(1,2)+psis{i}(2,2)) ];
    phiTemp = phisStar{i} ./ phis{i};
    psis{i+1} = repmat(phiTemp', 1, 2).*psis{i+1};    
end

%Reverse Pass (Distribute phase)
for i = (n-2):-1:1
    phisDoubleStar{i} = [(psis{i+1}(1,1)+psis{i+1}(1,2)); (psis{i+1}(2,1)+psis{i+1}(2,2))];
    phiDoubleTemp = phisDoubleStar{i} ./ phisStar{i}';
    psis{i} = repmat(phiDoubleTemp',2,1).*psis{i}; 
end

%Normalizing psis
for i = 1:(n-1)
    sum = psis{i}(1,1) + psis{i}(1,2) + psis{i}(2,1) + psis{i}(2,2);      
    psis{i} = psis{i} / sum;
end

%Normalizing phis
for i = 1:(n-2)
    sum = phisDoubleStar{i}(1,1) + phisDoubleStar{i}(2,1);
    phisDoubleStar{i} = phisDoubleStar{i} / sum;    
end

disp('Psi Values:');
for i = 1:(n-1)
disp(psis{i}); 
end

disp('Phi Values:');
for i = 1:(n-2)
disp(phisDoubleStar{i}'); 
end