classdef solverHMIP
    properties
        options
        funcs
        problem
        visualize
        performance
    end
    
    methods
        function self=solverHMIP(varargin)
            tic
            possProperties_problem = {'problem','options'};
            k=1;
            while k<nargin
                switch varargin{k}
                    case possProperties_problem
                        self.(varargin{k})=varargin{k+1};
                        k=k+2;
                    otherwise
                        possibilities=strjoin(possProperties_problem,', ');
                        if ischar(varargin{k})
                            error(['Unknown property:', varargin{k},', possible problem specifications include: ',possibilities])
                        else
                            error(['Please check that your property specifictions are correct. Possible specifications include: ', possibilities])
                        end
                end
            end
            self.visualize=VisualizeHMIP;
        
            if isempty(self.options)
                self.options=OptionsHMIP;
            end
            
            self.funcs=functionsHMIP;
            self.funcs.size=self.problem.size;
            self.funcs.activation_type=self.options.activation_type;
            
            if isempty(self.options.beta)~=1
                self.funcs.beta=self.options.beta;
            elseif isempty(self.options.beta)==1 && (strcmp(self.options.direction_method,'binary')||strcmp(self.options.direction_method,'soft_binary'))
                self.funcs.beta=1;
                self.options.beta=1;
            else
                self.options.beta=10*self.problem.binary_index+(1-self.problem.binary_index);
                self.funcs.beta=self.options.beta;
            end

            if isempty(self.problem.smoothness_coefficient)
                self.problem.smoothness_coefficient=self.funcs.compute_approximate_smoothness_coef(self.problem.gradient,self.problem.lb,self.problem.ub,self.problem.size);
                self.problem.step_size_smoothness=1/self.problem.smoothness_coefficient;
            else
                self.problem.step_size_smoothness=1/self.problem.smoothness_coefficient;
            end
            
            if strcmp(self.options.initial_ascent_method,'none')==0 && isempty(self.options.initial_ascent_stop_val)
                self.options.initial_ascent_stop_val=(self.problem.ub-self.problem.lb)/10^2;
            end
            
            if length(self.problem.lb)==1
                self.problem.lb=self.problem.lb*ones(self.problem.size,1);
            end
            
            if length(self.problem.ub)==1
                self.problem.ub=self.problem.ub*ones(self.problem.size,1);
            end
            
            self.performance=struct('binary_cv',0);
        end
        
        % main hopfield update
        function [x_h,x,fval,self]=main_hopfield(self)
            if self.options.keep_hopfield_trajectory==0
                iter=1;
                x=self.initialization;
                x_previous=x+1;
                x_h=self.funcs.inverse_activation(x,self.problem.lb,self.problem.ub);
                direction=1;
                while iter<self.options.num_iterations_max && norm(direction)>10^-6 && norm(x-x_previous)>10^-6
                    x_previous=x;
                    gradient=self.compute_gradient(x);
                    binary_convergence=self.binary_convergence_test(x);
                    direction=self.get_direction(x,gradient);
                    step=self.get_step_size(x, direction, gradient);
                    if binary_convergence~=0
                        [x_h,x]=hopfield_update(self,x_h,direction,step);
                    else
                        x=self.projected_gradient_update(x,direction,self.problem.step_size_smoothness);
                    end
                    iter=iter+1;
                end
                self=binary_cv(self,x);
                fval=self.problem.objective(x);
            elseif self.options.keep_hopfield_trajectory==1
                iter=1;
                x=[self.initialization,NaN*zeros(self.problem.size,self.options.num_iterations_max-1)];
                x_previous=x(:,1)+1;
                x_h=[self.funcs.inverse_activation(x,self.problem.lb,self.problem.ub),NaN*zeros(self.problem.size,self.options.num_iterations_max-1)];
                fval=NaN*zeros(self.options.num_iterations_max+1,1);
                fval(1)=self.problem.objective(x(:,1));
                direction=1;
                while iter<self.options.num_iterations_max && norm(direction)>10^-6 && norm(x(:,iter)-x_previous)>10^-6
                    x_previous=x(:,iter);
                    gradient=self.compute_gradient(x(:,iter));
                    binary_convergence=self.binary_convergence_test(x(:,iter));
                    if binary_convergence~=0
                        direction=self.get_direction(x(:,iter),gradient);
                        step=self.get_step_size(x(:,iter), direction, gradient);
                        [x_h(:,iter+1),x(:,iter+1)]=hopfield_update(self,x_h(:,iter),direction,step);
                    else
                        direction=-(1-self.problem.binary_index).*gradient;
                        x(:,iter+1)=self.projected_gradient_update(x(:,iter),direction,self.problem.step_size_smoothness);
                    end
                    iter=iter+1; 
                    fval(iter)=self.problem.objective(x(:,iter));
                end
                self=binary_cv(self,x(:,iter-1));
            end
        end
        
        % hopfield update
        function [x_h,x]=hopfield_update(self,x_h,direction,step)
            x_h=x_h+step*direction;
            x=self.funcs.activation(x_h,self.problem.lb,self.problem.ub);
        end
        
        % projected gradient descent
        function x=projected_gradient_update(self,x,direction,step)
            x=x+step*direction;
            x=self.funcs.box_projection(x,self.problem.lb,self.problem.ub);
        end
        
        % compute gradient
        function gradient=compute_gradient(self,x)
            gradient=self.problem.gradient(x);
        end
        
        % know if all binary index converged to binary selfution
        function out=binary_convergence_test(self, x)
            lb=self.problem.lb;
            ub=self.problem.ub;
            b_index=self.problem.binary_index;
            out =norm(1*(b_index.*(x-lb).*(ub-x)~=0));
        end
              
        % get the direction
        function direction=get_direction(self,x,gradient)
            lb=self.problem.lb;
            ub=self.problem.ub;
            av=0.5*(lb+ub);
            b_index=self.problem.binary_index;
            
            if strcmp(self.options.direction_method,'gradient')
                direction=-self.funcs.normalize(gradient);
            end
            
            if strcmp(self.options.direction_method,'stochastic')
                direction=-self.funcs.normalize(gradient.*round(rand(self.problem.size,1)));
            end
            
            if  strcmp(self.options.direction_method,'binary')||strcmp(self.options.direction_method,'soft_binary')
                g=-self.funcs.proxy_distance_activation(x,lb,ub).*gradient;
                g= self.funcs.normalize(g);
                h=-self.funcs.normalize(gradient);
                gamma=self.options.gamma;
                theta=self.options.theta;
                % define b
                if strcmp(self.options.direction_method,'binary')
                    b=self.problem.binary_index.*self.sign_rnd(x-av).*(b_index.*(ub-x).*(lb-x)~=0);
                elseif strcmp(self.options.direction_method,'soft_binary')
                    b=self.problem.binary_index.*self.funcs.activation((x-av).*(ub-lb)).*(b_index.*(ub-x).*(lb-x)~=0);
                end
                b=self.funcs.normalize(b);
                w=gamma*b+(1-gamma)*h;
                y=max(0,-g'*w+cot(theta)*sqrt(norm(w)^2-(g'*w)^2));
                direction=self.funcs.normalize(w+y*g);
                
            end
        end
       
        
        % get the step size
        function step=get_step_size(self,x, direction, gradient)
            if strcmp(self.options.step_size_method,'classic')
                numerator=-(self.funcs.proxy_distance_activation(x,self.problem.lb,self.problem.ub).*gradient)'*direction;
                denominator=norm(self.funcs.beta.*direction)^2+12*(self.funcs.beta.^2.*direction.^2)'*abs(gradient);
                if denominator~=0
                    step=numerator/denominator;
                else
                    step=0;
                end
            end
            if strcmp(self.options.step_size_method,'constant')
                if isempty(self.options.step_size_val)
                    step=self.problem.step_size_smoothness;
                else
                    step=self.options.step_size_val;
                end
            end
        end
        
        % initialization
        function x_0=initialization(self)
            if isempty(self.problem.x_0)
                if length(self.problem.ub)==self.problem.size
                    x_0=0.5*(self.problem.ub+self.problem.lb);
                elseif length(self.problem.ub)==1
                    x_0=0.5*(self.problem.ub+self.problem.lb)*ones(self.problem.size,1);
                end
            end
            if strcmp(self.options.initial_ascent_method,'binary_neutral_ascent')||strcmp(self.options.initial_ascent_method,'ascent')
                direction=1;
                iter=1;
                while iter<1000 && norm(direction)>10^-6 && self.funcs.is_in_box(x_0,self.problem.lb+self.options.initial_ascent_stop_val,self.problem.ub-self.options.initial_ascent_stop_val)
                    gradient=self.compute_gradient(x_0);
                    if norm((1-self.problem.binary_index).*gradient)==0
                        gradient=self.funcs.normalize(rand(self.problem.size,1)-0.5);
                    end
                    if strcmp(self.options.initial_ascent_method,'binary_neutral_ascent')
                        direction=(1-self.problem.binary_index).*gradient;
                    else
                        direction=gradient;
                    end
                    x_0=self.funcs.box_projection(x_0+self.problem.step_size_smoothness*direction,self.problem.lb,self.problem.ub);
                    iter=iter+1;
                end
                x_0=self.funcs.box_projection(x_0,self.problem.lb+self.options.initial_ascent_stop_val,self.problem.ub-self.options.initial_ascent_stop_val);
            end
        end
        
        % compute f_val at optimum with local quadratic approximation
        function fval=compute_f_val_approx(self)
            n=self.problem.size;
            A=self.problem.A;
            b=self.problem.b;
            Aeq=self.problem.Aeq;
            beq=self.problem.beq;
            lb=self.problem.lb.*ones(n,1);
            ub=self.problem.ub.*ones(n,1);
            penalty_eq=self.options.penalty_eq;
            penalty_ineq=self.options.penalty_ineq;
            if norm(Aeq,'fro')==0 && norm(beq)==0
                n_eq=0;
            else
                n_eq=size(Aeq,1);
            end
            if norm(A,'fro')==0 && norm(b)==0
                n_ineq=0;
            else
                n_ineq=size(A,1);
            end
            x_0=0.5*(lb+ub);
            [H,f,e]=self.funcs.quadratic_model(self.problem.objective,self.problem.gradient,x_0,norm(ub-lb)*10^(-3));
            if n_ineq>0
                H=[H+penalty_eq*(Aeq'*Aeq),zeros(n,n_ineq);zeros(n_ineq,n),penalty_ineq*eye(n_ineq)];
            else
                H=H+penalty_eq*(Aeq'*Aeq);
            end
            if n_ineq>0
                f=[f-penalty_eq*Aeq'*beq;zeros(n_ineq,1)];
            else
                f=f-penalty_eq*Aeq'*beq;
            end
            if n_ineq>0 && n_eq>0
                Aequa=[A,-eye(n_ineq);Aeq,zeros(n_eq,n_ineq)];
                bequa=[b;beq];
            elseif n_ineq==0 && n_ineq>0
                Aequa=A_eq;
                bequa=beq;
            elseif n_ineq>0 && n_ineq==0
                Aequa=[A,-eye(n_ineq)];
                bequa=b;
            elseif n_ineq==0 && n_eq==0
                Aequa=[];
                bequa=[];
            end
            lb=[lb;-10^6*ones(n_ineq,1)];
            ub=[ub;zeros(n_ineq,1)];
            options_loc = optimoptions('quadprog','Display','off');
            [~,fval]=quadprog(H,f,[],[],Aequa,bequa,lb,ub,[],options_loc);
            fval=round(fval+e+0.5*penalty_eq*norm(beq)^2,6);
        end
        
        %% get the dual variables
        function [x,dual_val_eq,dual_val_ineq]=dual_variables_problem(self)
            n=self.problem.size;
            A=self.problem.A;
            b=self.problem.b;
            Aeq=self.problem.Aeq;
            beq=self.problem.beq;
            lb=self.problem.lb;
            ub=self.problem.ub;
            precision=10^-4;
            if isempty(Aeq)~=1
                n_eq=size(Aeq,1);
            end
            if isempty(A)~=1
                n_ineq=size(A,1);
            end
            % surrogate lagrangian
            if strcmp(self.options.dual_method,'surrogate_lagrangian')
                % case with both equality and inequality constraints
                if isempty(Aeq)~=1 && isempty(A)~=1
                    dual_val_eq=zeros(n_eq,1);
                    previous_dual_val_eq=dual_val_eq+1;
                    dual_val_ineq=zeros(n_ineq,1);
                    previous_dual_val_ineq=dual_val_ineq+1;
                    if length(lb)==n
                        x=0.5*(lb+ub);
                    else
                        x=0.5*(lb+ub)*ones(n,1);
                    end
                    iter=1;
                    M_eq=norm(beq)+sum(max(abs(lb),abs(ub)).^2)*norm(Aeq);
                    M_ineq=norm(b)+sum(max(abs(lb),abs(ub)).^2)*norm(A);
                    q=self.compute_f_val_approx;
                    [~,L]=self.dual_function(x,dual_val_ineq,dual_val_eq);
                    past_steps=3;
                    c_eq_past=zeros(past_steps,1);
                    c_eq=abs(q-L)/norm(Aeq*x-beq)^2;
                    c_eq_past(1)=c_eq;
                    c_ineq=abs(q-L)/norm(A*x-b)^2;
                    while iter<10^4 &&( norm(previous_dual_val_eq-dual_val_eq,'Inf')>precision || norm(previous_dual_val_ineq-dual_val_ineq,'Inf')>precision ) && (norm(Aeq*x-beq,'Inf')>precision || norm(max(0,A*x-b),'Inf'))
                        previous_dual_val_eq=dual_val_eq;
                        previous_dual_val_ineq=dual_val_ineq;
                        previous_x=x;
                        x=self.dual_function(x,dual_val_ineq,dual_val_eq);
                        p=1-(1/iter)^0.95;
                        alpha_eq=1-(1/(M_eq*iter^p));
                        alpha_ineq=1-(1/(M_ineq*iter^p));
                        if norm(Aeq*x-beq)>0
                            c_eq_new=alpha_eq*c_eq*norm(Aeq*previous_x-beq)/norm(Aeq*x-beq);
                            c_eq=max(c_eq/100,min(c_eq_new,100*c_eq));
                        else
                            c_eq=0;
                        end
                        c_eq_past(mod(iter,past_steps)+1)=c_eq;
                        
                        if norm(A*x-b)>0
                            c_ineq=alpha_ineq*c_ineq*norm(A*previous_x-b)/norm(A*x-b);
                        else
                            c_ineq=0;
                        end
                        step_eq=(1/min(past_steps,iter))*sum(c_eq_past);
                        step_ineq=c_ineq;
                        dual_val_eq=dual_val_eq+step_eq*(Aeq*x-beq);
                        if norm(max(0,A*x-b))>0
                            dual_val_ineq=max(0,dual_val_ineq+step_ineq*(A*x-b));
                        else
                            dual_val_ineq=0;
                        end
                        disp(['step_eq: ',num2str(step_eq),' dual_val_eq',num2str(dual_val_eq) ])
                        disp(['dual_val_ineq',num2str(dual_val_ineq) ])
                        iter=iter+1;
                    end
                end
                disp(['Number of iterations to solve dual problem is: ', num2str(iter)])
            end
            % fmincon
            if strcmp(self.options.dual_method,'fmincon')
                options_fmincon = optimoptions('fmincon','SpecifyObjectiveGradient',true);
                fun=@(x)self.augmented_objective(x);
                x=fmincon(fun,0.5*(lb+ub).*ones(n,1),A,b,Aeq,beq,lb.*ones(n,1),ub.*ones(n,1),[],options_fmincon);
                x=round(x,6);
                [~,grad]=self.augmented_objective(x);
                
                fun=@(z)local_func_1(z,n);
                Ae=[-eye(n),Aeq',A',-eye(n),eye(n);   zeros(1,n+n_eq+n_ineq),(lb-x)',zeros(1,n);   zeros(1,2*n+n_eq+n_ineq),(ub-x)';   zeros(1,n+n_eq),(A*x-b)',zeros(1,2*n)];
                be=[-grad;0;0;0];
                lower=[-Inf*ones(n+n_eq,1);zeros(2*n+n_ineq,1)];
                upper=Inf*ones(3*n+n_eq+n_ineq,1);
                dual_vals=fmincon(fun,zeros(3*n+n_eq+n_ineq,1),[],[],Ae,be,lower,upper,[],options_fmincon);
                dual_val_eq=dual_vals(n+1:n+n_eq);
                dual_val_ineq=dual_vals(n+n_eq+1:n+n_eq+n_ineq);
            end
            
            function [f,g]= local_func_1(z,n)
                f=0.5*norm(z(1:n),2)^2;
                if nargout==2
                    g=[z(1:n);zeros(length(z)-n,1)];
                end
            end
        end
        
        %%
        function [x,Lagrange_val]=dual_function(self,dual_val_ineq,dual_val_eq)
            n=self.problem.size;
            lb=self.problem.lb;
            ub=self.problem.ub;
            options_fmincon = optimoptions('fmincon','SpecifyObjectiveGradient',true);
            fun=@(x)self.lagrangian(x,dual_val_ineq,dual_val_eq);
            x=fmincon(fun,0.5*(lb+ub).*ones(n,1),[],[],[],[],lb.*ones(n,1),ub.*ones(n,1),[],options_fmincon);
            if nargout==2
                Lagrange_val=fun(x);
            end
        end
        
        function [f,g]=lagrangian(self,x,dual_val_ineq,dual_val_eq)
            A=self.problem.A;
            b=self.problem.b;
            Aeq=self.problem.Aeq;
            beq=self.problem.beq;
            penalty_eq=self.options.penalty_eq;
            penalty_ineq=self.options.penalty_ineq;
            f=self.problem.objective(x)+0.5*penalty_ineq*norm(max(0,A*x-b))^2+0.5*penalty_eq*norm(Aeq*x-beq)^2+dual_val_ineq'*(A*x-b)+dual_val_eq'*(Aeq*x-beq);
            if nargout==2
                g=self.problem.gradient(x)+penalty_ineq*A'*max(0,A*x-b)+penalty_eq*Aeq'*(Aeq*x-beq)+A'*dual_val_ineq+Aeq'*dual_val_eq;
            end
        end
        
        function [f,g]=augmented_objective(self,x)
            A=self.problem.A;
            b=self.problem.b;
            Aeq=self.problem.Aeq;
            beq=self.problem.beq;
            penalty_eq=self.options.penalty_eq;
            penalty_ineq=self.options.penalty_ineq;
            f=self.problem.objective(x)+0.5*penalty_ineq*norm(max(0,A*x-b))^2+0.5*penalty_eq*norm(Aeq*x-beq)^2;
            if nargout>1
                g=self.problem.gradient(x)+penalty_ineq*A'*max(0,A*x-b)+penalty_eq*Aeq'*(Aeq*x-beq);
            end
        end
        
        function self=binary_cv(self,x)
            z=x.*self.problem.binary_index;
            out=sum(z.*(1-z));
            out=out/sum(self.problem.binary_index);
            disp(out)
            self.performance.binary_cv=out;
        end
        

    end
    
    methods(Static)
        function out=sign_rnd(x)
            out=sign(x);
            out=out+(out==0).*sign(rand(size(x))-0.5);
        end
    end
end
