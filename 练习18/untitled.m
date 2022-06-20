W1=randn(9,9,20);
W2=(2*rand(100,2000)-1)/20;
W3=(2*rand(10,100)-1)/10;
t1 = clock;
for epoch=1:3
    [W1,W2,W3]=CNN(W1,W2,W3,X_Train,D_Train);
end
t2 = clock;
fprintf('训练所需时间: %f min\n',etime(t2, t1)/60);