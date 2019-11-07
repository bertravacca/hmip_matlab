classdef OptionsHMIP
        properties
        step_size_val
        step_size_method='classic'
        direction_method='gradient'
        activation_type='sin'
        beta
        theta=pi/2-0.01
        gamma=0.95
        num_iterations_max=1000
        stopping_criterion
        precision_stopping_criterion
        initial_ascent_method='binary_neutral_ascent'
        initial_ascent_stop_val
        keep_hopfield_trajectory=0
        penalty_eq=1;
        penalty_ineq=1;
        dual_method='fmincon'
    end
    
    methods
        function options= OptionsHMIP(varargin)
            possProperties = {'step_size_val','step_size_method','direction_method','activation_type',...
                'beta','theta','gamma','num_iterations_max','keep_hopfield_trajectory'...
                'stopping_criterion','precision_stopping_criterion','initial_ascent_method', 'initial_ascent_stop_criterion','penlty_eq','penalty_ineq','dual_method'};
            k=1;
            while k<nargin
                switch varargin{k}
                    case possProperties
                        options.(varargin{k})=varargin{k+1};
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
            
            poss_direction_method={'gradient','stochastic','soft_binary','binary'};
            switch options.direction_method
                case poss_direction_method
                otherwise
                    possibilities=strjoin(poss_direction_method,', ');
                    error(['Please check that your property specifictions are correct for direction_method. Possible specifications include: ', possibilities])
            end
            
            poss_step_size_method={'classic','constant'};
            switch options.step_size_method
                case poss_step_size_method
                otherwise
                    possibilities=strjoin(poss_step_size_method,', ');
                    error(['Please check that your property specifictions are correct for step_size_method Possible specifications include: ', possibilities])
            end
            
            poss_initial_ascent_method={'none','ascent','binary_neutral_ascent'};
            switch options.initial_ascent_method
                case poss_initial_ascent_method
                otherwise
                    possibilities=strjoin(poss_initial_ascent_method,', ');
                    error(['Please check that your property specifictions are correct for initial_ascent_method Possible specifications include: ', possibilities])
            end
            

            
        end
    end
end

