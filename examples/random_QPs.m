addpath('/Applications/CPLEX_Studio128/cplex/matlab/x86-64_osx/')
clear all; close all;
% addpath to solver
str=strsplit(pwd,'/');
addpath(char(join(str(1:length(str)-1),'/')));

% define problem parameters at random
n = 20;
[binary_indicator,Q,q]=random_qp_parameters('no_constraints',n,0.4);
lb=0;
ub=1;

% hmip solver
objective=@(x) 0.5*x'*Q*x+q'*x;
gradient=@(x) Q*x+q;
problem=problemHMIP('objective',objective,'gradient',gradient,'size',n,'binary_index',binary_indicator,'lb',lb,'ub',ub);
options=OptionsHMIP('num_iterations_max',10^4,'keep_hopfield_trajectory',1,'activation_type','tanh','direction_method','binary');
solver=solverHMIP('problem',problem,'options',options);
solver=solver.main_hopfield;
% brute force hopfield
[x_bf,fval_bf,step_size_bf]=solver.brute_force_hopfield;
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
if isnan(solver.performance.transition_iter) == 0
    xline(solver.performance.transition_iter, 'b--')
end
semilogx(fval_bf,'g')
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



