classdef testsHMIP
    properties
        pass
        num_tests=1
        funcs=functionsHMIP
    end
    methods
        function tests=testsHMIP()
            out_1=tests.test_all_activations;
            out_2=tests.test_all_inverse_activations;
            out_3=tests.test_quadratic_model_bfgs;
            tests.pass=out_1*out_2*out_3;
            if tests.pass==1
                disp('all tests were succesful')
            end
        end
        
        function out=test_all_activations(tests)
            tests.funcs.activation_type='pwl';
            out_1=tests.test_activation;
            if out_1==0
                error('pwl fails tests')
            end
            tests.funcs.activation_type='sin';
            out_2=tests.test_activation;
            if out_2==0
                error('sin fails tests')
            end
            tests.funcs.activation_type='tanh';
            out_3=tests.test_activation;
            if out_3==0
                error('tanh fails tests')
            end
            out=out_1*out_2*out_3;
        end
        
        function out=test_all_inverse_activations(tests)
            tests.funcs.activation_type='pwl';
            out_1=tests.test_inverse_activation;
            if out_1==0
                error('pwl fails inverse test')
            end
            tests.funcs.activation_type='sin';
            out_2=tests.test_inverse_activation;
            if out_2==0
                error('sin fails inverse test')
            end
            tests.funcs.activation_type='tanh';
            out_3=tests.test_inverse_activation;
            if out_3==0
                error('tanh fails inverse test')
            end
            out=out_1*out_2*out_3;
        end
        
        function out=test_activation(tests)
            out=NaN*zeros(10,1);
            for k=1:10
                tests.funcs.beta=k;
                out_1=(tests.funcs.activation(0.5)==0.5);
                out_2=(max(tests.funcs.activation(-100:1:100))<=1);
                out_3=(min(tests.funcs.activation(-100:1:100))>=0);
                x_1=0.5:100+0.5;
                x_2=flip(-100+0.5:0.5);
                f_1=tests.funcs.activation(x_1);
                f_2=1-tests.funcs.activation(x_2);
                out_4=(f_1==f_2);
                if out_1*out_2*out_3*out_4==1
                    out(k)=true;
                else
                    out(k)=false;
                end
            end
            out=prod(out);
        end
        
        function out=test_inverse_activation(tests)
            x=rand(100,1);
            f_1=tests.funcs.activation(x);
            f_2=tests.funcs.inverse_activation(f_1);
            if max(abs(x-f_2))<10^-10
                out=true;
            else
                out=false;
            end
        end
        
        function out=test_quadratic_model_bfgs(tests)
            n=100;
            Q=rand(n,n);
            Q=Q'*Q;
            lam=min(eig(Q));
            Q=Q+(10^-2-lam)*eye(n);
            q=rand(n,1);
            objective=@(x) 0.5*x'*Q*x+q'*x;
            gradient=@(x) Q*x+q;
            disp('solver'); disp(toc)
            [~,~,~,x,~]=tests.funcs.quadratic_model_bfgs(objective,gradient,0.5);
            options = optimoptions('quadprog','Display','off');
            x_quad=quadprog(Q,q,[],[],[],[],[],[],[],options);
            out=(abs(objective(x_quad)-objective(x))<10^(-5))  ;
        end
        
        function out=test_quadratic_model(tests)
            n=100;
            Q=rand(n,n);
            Q=Q'*Q;
            lam=min(eig(Q));
            Q=Q+(10^-2-lam)*eye(n);
            q=rand(n,1);
            objective=@(x) 0.5*x'*Q*x+q'*x;
            gradient=@(x) Q*x+q;
            disp('solver'); disp(toc)
            [H,f,cst]=solver.funcs.quadratic_model(objective,gradient,0.5,0.1);
            out=(norm(H-Q,'fro')<10^(-6))*(norm(q-f)<10^(-6))*(abs(cst)<10^(-6));
        end
    end
    
    methods(Static)

    end
end