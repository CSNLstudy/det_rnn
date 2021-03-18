% synchrony

function synchrony()

A = [2, -1.52; 1.52, -0.5]; %one neural subpopulation with close eig~1
gain = 1;
net.poisson     = 0;

%% fig 2: no bias
%{
% no input
Wrnn            = blkdiag(A,A,0);
net.Wrnn        = Wrnn; 
net.alpha       = [0.6, 0.6, 0.6, 0.6, 0.6];
net.b           = [0,0,0,0,0];
net = make_network(net);

r1 = plot_network(net,[10,0,10,0,0]);

% input only network
Wrnn            = blkdiag(A,A,0);
Wrnn(:,end)     = [0;0;0;0;0]; % output weights
Wrnn(end,:)     = [1,0,1,0,0]; % input weights
net.Wrnn        = Wrnn; 
net = make_network(net);

r2 = plot_network(net,[5,5,0,0,0]); % desynchronized
r3 = plot_network(net,[5,5,5,5,0]); % synchronized

figure; 
errorbar([1,2,3], ...
    [mean(r1(500:end,5)), mean(r2(500:end,5)), mean(r3(500:end,5))],...
    [std(r1(500:end,5)), std(r2(500:end,5)), std(r3(500:end,5))],...
    'CapSize',20,'LineWidth',2,'LineStyle','none');
xticks([1,2,3]);
xticklabels({'no input','desynchronized input','synchronized input'});
ylabel('mean activity'); title('modulatory neuron');

%% fig3: with bias
% no input network

Wrnn            = blkdiag(A,A,0);
net.Wrnn        = Wrnn; 
net.alpha       = [0.6, 0.6, 0.6, 0.6, 0.6];
net.b           = [5,0,5,0,0];
net = make_network(net);

r1 = plot_network(net,[10,0,10,0,0]);

% input only network
Wrnn            = blkdiag(A,A,0);
Wrnn(:,end)     = [0;0;0;0;0]; % output weights
Wrnn(end,:)     = [1,0,1,0,0]; % input weights

net.Wrnn        = Wrnn; 
net = make_network(net);

r2 = plot_network(net,[5,5,0,0,0]); % desynchronized
r3 = plot_network(net,[5,5,5,5,0]); % synchronized

figure; 
errorbar([1,2,3], ...
    [mean(r1(500:end,5)), mean(r2(500:end,5)), mean(r3(500:end,5))],...
    [std(r1(500:end,5)), std(r2(500:end,5)), std(r3(500:end,5))],...
    'CapSize',20,'LineWidth',2,'LineStyle','none');
xticks([1,2,3]);
xticklabels({'no input','desynchronized input','synchronized input'});
ylabel('mean activity'); title('modulatory neuron');

%}

%% fig4: self-inhibition
% no input network
net.alpha       = [0.6, 0.6, 0.6, 0.6, 0.3];
%net.b           = [0,0,0,0,-1];
%net.b           = [5,0,5,0,0];
net.b           = [5,0,5,0,-20];


Wrnn            = blkdiag(A,A,0);
%Wrnn(:,end)     = [1;0;1;0;0]; % output weights
Wrnn(end,:)     = [1,0,1,0,0]; % input weights
net.Wrnn        = gain*Wrnn; 
net             = make_network(net);

corrcoefflist = [];
meanlist      = [];
stdlist       = [];

for iter = 1:20   
    init = rand(1,net.nneurons);
    r = plot_network(net,init); 
    R = corrcoef(r(500:end,1)',r(500:end,3)');
    corrcoefflist(end+1) = R(2,1);
    meanlist(end+1) = mean(r(500:end,5));
    stdlist(end+1) = std(r(500:end,5));
    close(gcf)
end
figure('Position',[2141 389 671 545]); 
errorbar(corrcoefflist, meanlist, stdlist,...
    'CapSize',20,'LineWidth',2,'LineStyle','none');
xlabel('Corrcoeff')
ylabel('mean activity'); title('modulatory neuron');

%% fig5: add feedback
net.alpha       = [0.6, 0.6, 0.6, 0.6, 0.2];
net.b           = [5,0,5,0,0];

Wrnn            = blkdiag(A,A,0);
net.Wrnn        = Wrnn; 
net = make_network(net);
r1 = plot_network(net,[1,0,1,0,0]);

end

function net = make_network(net)
    net.nneurons    = size(net.Wrnn,1);
    net.decay       = diag(1-net.alpha) * eye(net.nneurons);
    net.NoiseStd    = 0.1;
    net.W           = net.decay + diag(net.alpha) * net.Wrnn;

    [S,D] = eig(net.W);
end

function r = plot_network(net, init)
    if exist('init','var')
        r(1,:)  = init;
    else
        r(1,:)  = rand(1,net.nneurons);
    end
    r = propagate_dynamics(net, init);

    figure('Position',[2141 389 671 545])
    % initial conditions
    subplot(3,3,1); 
    imagesc(net.Wrnn);colorbar;title('Wrnn');caxis([-2,2]);
    
    subplot(3,3,4); 
    imagesc(net.W);colorbar;title('W');caxis([-2,2]);
    
    subplot(3,3,7); 
    imagesc(init');colorbar;title('initial conditions');
    
    ax = subplot(3,3,2:3);hold on;
    plot(r(:,1), 'b');
    plot(r(:,2), 'r');
    title('population 1'); legend({'exc', 'inh'}); xlabel('frames');

    ax1 = subplot(3,3,5:6); hold on;
    plot(r(:,3), 'b');
    plot(r(:,4), 'r');
    title('population 2'); legend({'exc', 'inh'}); xlabel('frames');

    ax2 = subplot(3,3,8:9); hold on;
    plot(r(:,5), 'b');
    title('modulation'); legend({'exc'}); xlabel('frames');
    
    linkaxes([ax,ax1,ax2],'x');
end

function r = propagate_dynamics(net, init)
    Niters = 1000;
    
    if exist('init','var')
        r(1,:)  = init;
    else
        r(1,:)  = rand(1,net.nneurons);
    end
    
    for iter = 2:Niters
        if net.poisson == 1
            r(iter,:) = net.decay*r(iter-1,:)' + ...
                diag(net.alpha)*relu(net.Wrnn*r(iter-1,:)'+ ...
                    r(iter-1,:)'.* (net.NoiseStd * diag(sqrt(2./net.alpha))*rand(net.nneurons,1)) +...
                    net.b');
        else
            r(iter,:) = net.decay*r(iter-1,:)' + ...
                diag(net.alpha)*relu(net.Wrnn*r(iter-1,:)'+ ...
                    net.NoiseStd * diag(sqrt(2./net.alpha))*rand(net.nneurons,1) +...
                    net.b');
        end
    end
end

function x =relu(x)
    x(x<0) = 0;
end