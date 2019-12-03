function [binary_indicator,Q,q,A,b,Aeq,beq]=random_qp_parameters(type,n,sparsity)
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
q_tmp=(1/n)*mvnrnd(zeros(n,1),sigma)'; clear B sigma
x_max = 10^6*ones(n,1); x_min = -10^6*ones(n,1);
for k = 1:n
    if binary_indicator(k) == 1 && rand()>0.3
        v_1 = rand(); v_2 = rand();
        x_max(k) = max(v_1, v_2);
        x_min(k) = min(v_1, v_2);
    end
end
rho = 1;
cvx_begin quiet
variable x(n,1)
variable q(n,1)
variable t(1)
minimize(rho*norm(q-q_tmp,2))
x<=x_max
x>=x_min
Q*x+q == 0
disp(x)
cvx_end

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