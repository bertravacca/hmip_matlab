classdef functionsHMIP
    properties
        activation_type='sin'
        beta=1;
        size=1;
    end
    
    methods
        function funcs=functionsHMIP
            if funcs.beta==1
                funcs.beta=ones(funcs.size,1);
            end
        end

        function out=activation(funcs,x,lb,ub)
            if nargin==2
                lb=zeros(funcs.size,1);
                ub=ones(funcs.size,1);
            end
            y=(x-lb)./(ub-lb);
            if strcmp(funcs.activation_type,'tanh')
                out=0.5*(tanh(2*funcs.beta.*(y-0.5))+1);
            elseif strcmp(funcs.activation_type,'pwl')
                out=funcs.box_projection(funcs.beta.*(y-0.5)+0.5,0,1);
            elseif strcmp(funcs.activation_type,'sin')
                out=(0.5*sin(2*funcs.beta.*(y-0.5))+0.5).*(0.5-pi./(4*funcs.beta)<y).*(y<0.5+pi./(4 *funcs.beta))+(y>=0.5+pi./(4*funcs.beta));
            elseif strcmp(funcs.activation_type,'identity')
                out=y;
            end
            out=(ub-lb).*out+lb;
        end
        
        function out=proxy_distance_activation(funcs,x,lb,ub) 
            if nargin==2
                lb=zeros(funcs.size,1);
                ub=ones(funcs.size,1);
            end
            if strcmp(funcs.activation_type,'tanh')
                out=(4./((ub-lb).^2)).*funcs.beta.*(x-lb).*(ub-x);
            elseif strcmp(funcs.activation_type,'pwl')
                out=funcs.beta.*(x~=ub).*(x~=lb);
            elseif strcmp(funcs.activation_type,'sin')
                out=2./(ub-lb).*funcs.beta.*sqrt((x-lb).*(ub-x));
            elseif strcmp(funcs.activation_type,'identity')
                out=1;
            end
        end
        
        function out=inverse_activation(funcs,x,lb,ub)
            if nargin==2
                lb=zeros(funcs.size,1);
                ub=ones(funcs.size,1);
            end
            y=(x-lb)./(ub-lb);
            if strcmp(funcs.activation_type,'tanh')
                out=(1./(2*funcs.beta)).*atanh(2.*(y-0.5))+0.5;
            elseif strcmp(funcs.activation_type,'pwl')
                out=(1./funcs.beta).*(y-0.5)+0.5;
            elseif strcmp(funcs.activation_type,'sin')
                out=(1./(2*funcs.beta)).*asin(2.*(y-0.5))+0.5;
            elseif strcmp(funcs.activation_type,'identity')
                out=y;
            end
            out=(ub-lb).*out+lb;
        end
        
    end
    
    methods(Static)
        function [H,h,cst]=quadratic_model(fun,grad_fun,x_0,epsilon)
             n=length(grad_fun(x_0));
             x_0=x_0.*ones(n,1);
             E=eye(n);
             H=NaN*zeros(n,n);
             grad_0=grad_fun(x_0);
             for i=1:n
                 h=(1/epsilon)*(grad_fun(x_0+epsilon*E(:,i))-grad_0);
                 H(:,i)=h;
             end
             H=0.5*(H+H');
             m=min(eig(H));
             if m<0
                 H=H-m*eye(n);
             end
             h=grad_0-H*x_0;
             cst=fun(x_0)-0.5*x_0'*H*x_0-h'*x_0;
        end
        
        function [H,h,cst,x,f_val]=quadratic_model_bfgs(fun,grad_fun,x_0,num)
            if nargin==3
                num=10^3;
            end
            n=length(grad_fun(x_0));
            x=NaN*zeros(n,num+1);
            x(:,1)=x_0.*ones(n,1);
            delta_x=NaN*zeros(n,num);
            y=NaN*zeros(n,num);
            inv_H=0.01*eye(n);
            iter=1;
            var_x=1;
            gradient=grad_fun(x(:,1));
            new_gradient=gradient;
            while iter<num && norm(var_x,'Inf')>10^(-6) && norm(new_gradient)>10^-6
                gradient=new_gradient;
                delta_x(:,iter)=-0.5*inv_H*gradient;
                x(:,iter+1)=x(:,iter)+delta_x(:,iter);         
                new_gradient=grad_fun(x(:,iter+1));
                y(:,iter)=new_gradient-gradient;
                denominator=delta_x(:,iter)'*y(:,iter);
                if abs(denominator)>10^-9
                    inv_H=(eye(n)-delta_x(:,iter)*y(:,iter)'/denominator)*inv_H*(eye(n)-y(:,iter)*delta_x(:,iter)'/denominator)+delta_x(:,iter)*delta_x(:,iter)'/denominator;       
                end
                var_x=norm(delta_x(:,iter),'Inf');
                iter=iter+1;
            end
            x=x(:,iter);
            H=round(inv(inv_H+10^(-6)*eye(n)),6);
            h=round(new_gradient-H*x,6);
            cst=round(fun(x)-0.5*x'*H*x-h'*x,6);
            f_val=fun(x);
        end
        
        function out=box_projection(x,lb,ub)
            out=max(lb,min(x,ub));
        end
        
        function out=is_in_box(x,lb,ub)
            if max(lb,min(x,ub))==x
                out=true;
            else
                out=false;
            end
        end
        
        function smoothness_val=compute_approximate_smoothness_coef(gradient,lb,ub,n)
            n_rand=100*log(n);
            smoothness_val=0;
            for k=1:n_rand
                point_1=(ub-lb).*rand(n,1)+lb;
                point_2=(ub-lb).*rand(n,1)+lb;
                distance=norm(point_1-point_2);
                if distance>10^(-6)
                    smoothness_val=max(smoothness_val,norm(gradient(point_1)-gradient(point_2))/distance);
                end
            end
        end
        
        function z=normalize(x)
            if norm(x)>0
                z=x/norm(x);
            else
                z=zeros(length(x),1);
            end
        end
        
    end
end