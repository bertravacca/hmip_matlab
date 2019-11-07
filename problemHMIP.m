classdef problemHMIP
    properties
        objective
        gradient
        lb=0
        ub=1
        Aeq=0
        beq=0
        A=0
        b=0
        binary_index
        size
        smoothness_coefficient
        x_0
        step_size_smoothness
        max_svd_Aeq
        max_svd_A
    end
    
    methods
        function problem=problemHMIP(varargin)
            possProperties = {'objective','gradient','lb','ub','Aeq','A','beq','b','size','binary_index'};
            k=1;
            while k<nargin
                switch varargin{k}
                    case possProperties
                        problem.(varargin{k})=varargin{k+1};
                        k=k+2;
                    otherwise
                        possibilities=strjoin(possProperties,', ');
                        if ischar(varargin{k})
                            error(['Unknown property for OptionsFenchel class:', varargin{k},', possible specifications include: ',possibilities])
                        else
                            error(['Please check that your property specifictions are correct. Possible specifications include: ', possibilities])
                        end
                end
            end

            % make sure that the specified problem makes sense
            if isempty(problem.objective)||isempty(problem.gradient)
                error('The objective and gradient need to be specified')
            end
            
            if isempty(problem.size)
                error('Please specify the size if the problem')
            end
            
            % check that th objective is a scalar
            if length(problem.objective(rand(problem.size,1)))>1
                error('The objective must take scalar values')
            end
            
            if length(problem.gradient(rand(problem.size,1)))~=problem.size
                error('Gradient values must be of the same dimensions as the problem size')
            end
            
            if isempty(problem.binary_index)
                warning('You defined a problem without specifying binary constraints, by default the variables are all considered to be binary')
                problem.binary_index=ones(problem.size,1);
            end
            
            if length(problem.ub)>1&&problem.size~=length(problem.ub)
                error('ub must be a scalar or a vector of the same dimension as the problem size')
            end
            
            if length(problem.lb)>1&&problem.size~=length(problem.lb)
                error('ub must be a scalar or a vector of the same dimension as the problem size')
            end
            
            if sum(problem.lb>problem.ub)
                error('lb should be smaller than ub (componentwise)')
            end
            
            if sum(problem.binary_index.*(1-problem.binary_index))~=0
                error('binary index should be a vector consisting of 0 (continuous variables) and 1 (binary variables)')
            end
            
            if isempty(problem.Aeq)==0&&isempty(problem.beq)
                [m,n]=size(problem.Aeq);
                if n~=problem.size||(m~=length(problem.beq)&&m~=1)
                    error('incorrect dimension for matrix Aeq and/or vector beq')
                end
            elseif (isempty(problem.Aeq)==0&&isempty(problem.beq))||(isempty(problem.Aeq)&&isempty(problem.beq)==0)
                error('Please specify Aeq and beq')
            elseif max(size(problem.Aeq))>problem.size
                error('The number of equality constraints is higher than the problem size, either the problem is infeasible or some constraints are redundant')
            end
            
            if isempty(problem.A)==0&&isempty(problem.b)
                [m,n]=size(problem.A);
                if n~=problem.size||(m~=length(problem.b)&&m~=1)
                    error('incorrect dimension for matrix Aeq and/or vector beq')
                end
            elseif (isempty(problem.A)==0&&isempty(problem.b))||(isempty(problem.A)&&isempty(problem.b)==0)
                error('Please specify Aeq and beq')
            end
            
            if isempty(problem.smoothness_coefficient)==0
                problem.step_size_smoothness=1/problem.smoothness_coefficient;
            end
            
            problem.max_svd_Aeq=norm(problem.Aeq'*problem.Aeq);
            problem.max_svd_A=norm(problem.A'*problem.A);
      
        end
    end
end