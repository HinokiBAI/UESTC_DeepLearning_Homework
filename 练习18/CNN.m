function[W1,W2,W3]=CNN(W1,W2,W3,X_Train,D_Train)
   alpha=0.01;
   for k=1:60000
       X=X_Train(:,:,k);
       for n=1:20
       V1(:,:,n)=filter2(W1(:,:,n),X,'Valid');
       end
       Y1=ReLU(V1);
       Y2=(Y1(1:2:end,1:2:end,:)+Y1(2:2:end,1:2:end,:)+Y1(1:2:end,2:2:end,:)+Y1(2:2:end,2:2:end,:))/4;
       y2=reshape(Y2,[],1);
       v3=W2*y2;
       y3=ReLU(v3);
       v=W3*y3;
       y=Softmax(v);
       
       
       d=D_Train(:,k);
       e=d-y;
       delta=e;
       e3=W3'*delta;
       delta3=(v3>0).*e3;
       e2=W2'*delta3;
       E2=reshape(e2,size(Y2));
       E1=zeros(size(Y1));
       E2_4=E2/4;
       E1(1:2:end,1:2:end,:)=E2_4;
       E1(1:2:end,2:2:end,:)=E2_4;
       E1(2:2:end,1:2:end,:)=E2_4;
       E1(2:2:end,2:2:end,:)=E2_4;
       delta1=(V1>0).*E1;
       
       for n=1:20
       dW1(:,:,n)=alpha*filter2(delta1(:,:,n),X,'Valid');
       W1(:,:,n)=W1(:,:,n)+dW1(:,:,n);
       end
       W2=W2+alpha*delta3*y2';
       W3=W3+alpha*delta*y3';
   end
end