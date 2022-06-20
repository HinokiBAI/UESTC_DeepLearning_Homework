W1=randn(9,9,20);
W3=(2*rand(100,2000)-1)/20;
W4=(2*rand(10,100)-1)/10;
t1 = clock; 

for epoch=1:3
    [W1,W3,W4]=CNN(W1,W3,W4,X_Train,D_Train);
end
 
N=length(D_Test);
d_comp=zeros(1,N);
for k=1:N
    X=X_Test(:,:,k);
    for n=1:20
    V1(:,:,n)=filter2(W1(:,:,n),X,'Valid');
    end
    Y1=ReLU(V1);
    Y2=(Y1(1:2:end,1:2:end,:)+Y1(2:2:end,1:2:end,:)+Y1(1:2:end,2:2:end,:)+Y1(2:2:end,2:2:end,:))/4;
    y2=reshape(Y2,[],1);
    v3=W3*y2;
    y3=ReLU(v3);
    v=W4*y3;
    y=Softmax(v);
    [~,i]=max(y);
    d_comp(k)=i;   
end

t2 = clock;
[~,d_true]=max(D_Test);
acc=sum(d_comp==d_true);
fprintf('\nAccuracy: %f\n',acc/N);
fprintf('Time: %f min\n', etime(t2, t1)/60);
 
