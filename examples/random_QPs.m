addpath('/Applications/CPLEX_Studio128/cplex/matlab/x86-64_osx/')
clear all; close all;
% addpath to solver
str=strsplit(pwd,'/');
addpath(char(join(str(1:length(str)-1),'/')));

% define problem parameters at random
n=20;
beta=0.4;
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
gamma=0.8;
m_eq=floor(gamma*rand()*n);
m_ineq=floor(gamma*rand()*n);
Aeq=rand(m_eq,n).*(rand(m_eq,n)>density);
A=rand(m_ineq,n).*(rand(m_ineq,n)>density);
z=(1-binary_indicator).*rand(n,1)+binary_indicator.*(rand(n,1)>0.5);
beq=Aeq*z;
b=A*z+10^-3*rand(m_ineq,1);
lb=0;
ub=1;

% hmip solver
objective=@(x) 0.5*x'*Q*x+q'*x;
gradient=@(x) Q*x+q;
problem=problemHMIP('objective',objective,'gradient',gradient,'size',n,'binary_index',binary_indicator,'lb',lb,'ub',ub);
options=OptionsHMIP('num_iterations_max',10^3,'keep_hopfield_trajectory',1,'activation_type','pwl','direction_method','binary');
solver=solverHMIP('problem',problem,'options',options);
solver=solver.main_hopfield;
% projected gradient descent
[x_pgd,fval_pgd,step_size_pgd]=solver.projected_gradient_descent;
% quadprog
options = optimset('Display', 'off');
[x_qp,fval_qp]=quadprog(Q,q,[],[],[],[],lb*ones(n,1),ub*ones(n,1),[],options);
% cplexmiqp
for k=1:n
    if binary_indicator(k)==1
        ctype(k)='B';
    else
        ctype(k)='C';
    end
end
[x_cplx,fval_cplx]=cplexmiqp(Q, q, [], [], [], [],[], [], [], lb*ones(n,1), ub*ones(n,1), ctype);

% plot fval for the different methods
figure(1)
semilogx(solver.fval,'b')
hold on
semilogx(fval_pgd,'r')
semilogx(fval_qp*ones(max(length(solver.fval),length(fval_pgd)),1),'r--')
semilogx(fval_cplx*ones(max(length(solver.fval),length(fval_pgd)),1),'b--')
hold off
title('Objective function value across iterations')
xlabel('# of iterations')
ylabel('fval')

%figure(2)
%semilogx(100*solver.step_size,'b')
%hold on
%semilogx(step_size,'r')



