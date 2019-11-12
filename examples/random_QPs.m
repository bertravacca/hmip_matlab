% addpath to solver
str=strsplit(pwd,'/');
addpath(char(join(str(1:length(str)-1),'/')));

% define problem parameters at random
n=1000;
beta=0.4;
Z=rand(n,1);
binary_indicator=ones(n,1).*(Z<=beta); clear Z;
density=0.4;
A=rand(n,n).*(rand(n,n)>density);
V=orth(A);
D=diag(n*rand(n,1));
Q=V'*D*V; clear V D A 
Q=0.5*(Q'+Q);
B=rand(n,n).*(rand(n,n)>density);
sigma=B'*B;
q=mvnrnd(zeros(n,1),sigma)'; clear B sigma
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

% use hmip solver
objective=@(x) 0.5*x'*Q*x+q'*x;
gradient=@(x) Q*x+q;
problem=problemHMIP('objective',objective,'gradient',gradient,'size',n,'binary_index',binary_indicator,'lb',lb,'ub',ub);
options=OptionsHMIP('num_iterations_max',1000,'keep_hopfield_trajectory',1,'activation_type','pwl','direction_method','gradient');
solver=solverHMIP('problem',problem,'options',options);
[x_h,x,fval,solver]=solver.main_hopfield;
%x=quadprog(Q,q,A,b,Aeq,beq,zeros(n,1),ones(n,1));


