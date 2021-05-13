function [B1, B2] = COSDISH (dataset, codeLength)

    dataset.traindata=normZeroMean(dataset.traindata);
    dataset.traindata=normEqualVariance(dataset.traindata);
	N1 = length(dataset.indexTrain);
	X1 = dataset.traindata(dataset.indexTrain, :);
	N2 = length(dataset.indexTest);
	X2 = dataset.traindata(dataset.indexTest, :);
	
	q = codeLength;
	calcS = @(Sc) calcNeighbor2(dataset, dataset.indexTrain, dataset.indexTrain(Sc));	

    
    sc=q; 
	T_sto=10;T_alt=3;
    B=randn(N1,q)>0;
	
	
    NT = (X1' * X1 + 1 * eye(size(X1, 2))) \ X1';
	B=B*2-1;
	nB=B;
	
	%-----------------------------------------------------training---------------------------------
	for iter=1:T_sto
		Omega = randsample(N1, sc);
		
		
		Gamma = setdiff([1:N1],Omega);
		
		
		S=calcS(Omega);
		
		r=sum(S(:))/sum(1-S(:));
		s1=1;
		s0=s1*r;
		S=S*(s1-s0)+s0;
		
		
		
	
		S_g=S(Gamma,:);
		S_o=S(Omega,:);
		
	
		for t=1:T_alt
		
			for k=1:q
				Q=zeros(sc,sc);

				
				if k==1
					lq=-2*q*S_o;
				else
					lq=lq+2*nB(Omega,k-1)*nB(Omega,k-1)';
				end
				QQ=lq;
				QQ(1:sc+1:end)=0;
				Q=QQ;
				
				
				
				if k==1
					lp=q*S_g;
				else 
					lp=lp-B(Gamma,k-1)*nB(Omega,k-1)';
				end
				pp=-2*lp'*B(Gamma,k);
				p=pp;

				
				for i=1:sc
					p(i)=2*p(i);
					for l=1:sc
						p(i)=p(i)-2*(Q(i,l)+Q(l,i));
					end
				end
				
				Q=Q*4;
				
				
				Q=[Q,p/2;p'/2,0];
				 
                
                Q=Q/max(max(abs(Q)));
                lambda=10000;   			
			
                P=chol(lambda*eye(sc+1)-Q);  
                
                nB(Omega,k)=Solve(P)*2-1;
                
            end
			nB(Gamma,:)=ex_sign(S_g*nB(Omega,:),B(Gamma,:));
			B=nB;
		end

	end
	
	
	B1=B;
		
		
		
		
	W=NT*B1;
    
	B2=X2*W>0;
		
	
end



function B= ex_sign(a,b)
	idx=find(abs(a)<1e-5);
	a(idx)=b(idx);
	B=(a>0)*2-1;
end