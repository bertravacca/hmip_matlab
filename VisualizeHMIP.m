classdef VisualizeHMIP
    properties
        count_figure=1;
    end
    methods
        function visualize=VisualizeHMIP
        end
        function visualize=phase_portrait_2D(visualize,x,Q,q,lb,ub)
            n=length(x(isnan(x(1,:))~=1));
            disp(['# Hopfield iterations:',num2str(n)])
            figure(visualize.count_figure)
            if length(lb)==1
                lb=[lb;lb];
            end
            if length(ub)==1
                ub=[ub;ub];
            end
            [x1grid, x2grid] = meshgrid(linspace(lb(1)-0.25*(ub(1)-lb(1)),ub(1)+0.25*(ub(1)-lb(1)),25), linspace(lb(2)-0.25*(ub(2)-lb(2)),ub(2)+0.25*(ub(2)-lb(2)),25));
            fgrid = 0.5*Q(1,1)*x1grid.^2 + 0.5*Q(2,2)*x2grid.^2+Q(1,2)*x1grid.*x2grid+x1grid.*q(1)+x2grid.*q(2);
            contourf(x1grid,x2grid,fgrid,50);
            axis('equal')
            clear x1grid; clear x2grid; clear fgrid;
            hold on;
            plot([lb(1) lb(2)],[lb(1),ub(2)],'k-','LineWidth',3);
            plot([lb(1) ub(2)],[lb(1),lb(2)],'k-','LineWidth',3);
            plot([ub(1) lb(2)],[ub(1),ub(2)],'k-','LineWidth',3);
            plot([ub(1) ub(2)],[lb(1),ub(2)],'k-','LineWidth',3);
            xlim([lb(1)-0.25*(ub(1)-lb(1)),ub(1)+0.25*(ub(1)-lb(1))]);
            ylim([lb(2)-0.25*(ub(2)-lb(2)),ub(2)+0.25*(ub(2)-lb(2))]);
            plot(x(1,:),x(2,:),'g','LineWidth',2)
            plot(x(1,:),x(2,:),'g.','MarkerSize',10)
            xlim([lb(1)-0.25*(ub(1)-lb(1)),ub(1)+0.25*(ub(1)-lb(1))]);
            ylim([lb(2)-0.25*(ub(2)-lb(2)),ub(2)+0.25*(ub(2)-lb(2))]);
            set(gca,'FontSize',20);
            plot(x(1,n), x(2,n),'gh','MarkerSize',14);
            plot(x(1,1), x(2,1),'go','MarkerSize',14);
            hold off
            visualize.count_figure=visualize.count_figure+1;
        end
    end
end