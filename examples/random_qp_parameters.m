function [Q,q,A,b,Aeq,beq]=random_qp_parameters(type,n,sparsity,num_constraints)
beta=sparsity;
Z=rand(n,1);
binary_indicator=ones(n,1).*(Z<=beta); clear Z;
density=0.4;
A=rand(n,n).*(rand(n,n)>density);
V=orth(A);
D=diag(n*rand(n,1));
Q=V'*D*V; clear V D A
Q=(0.5/n)*(Q'+Q);
B=rand(n,n).*(rand(n,n)>density);
sigma=B'*B;
q=(1/n)*mvnrnd(zeros(n,1),sigma)'; clear B sigma
if strcmp(type,'linear,constraints')
    gamma=num_constraints/n;
    m_eq=floor(gamma*rand()*n);
    m_ineq=floor(gamma*rand()*n);
    Aeq=rand(m_eq,n).*(rand(m_eq,n)>density);
    A=rand(m_ineq,n).*(rand(m_ineq,n)>density);
    z=(1-binary_indicator).*rand(n,1)+binary_indicator.*(rand(n,1)>0.5);
    beq=Aeq*z;
    b=A*z+10^-3*rand(m_ineq,1);
end
end