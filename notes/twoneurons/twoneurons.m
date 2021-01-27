Nsamples = 10000;
B = [1,0;0,-1]; % Dale's law matrix

%% Eigenvalues of positive matrices and Dale's law.
rand_eig         = nan(Nsamples,2);
rand_eig_real    = [];
dale_rand_eig    = nan(Nsamples,2);
dale_rand_eig_r  = [];

for iter = 1:Nsamples
    rand_eig(iter,:) = eig(rand(2,2));          % uniformly distributed random numbers in (0,1)
    dale_rand_eig(iter,:) = eig(rand(2,2)*B);   % Dales law applied.
    
    rand_eig_real(iter) = isreal(rand_eig(iter,:));
    dale_rand_eig_r(iter)  = isreal(dale_rand_eig(iter,:));
end

figure('Position', [1 41 1920 962]);
subplot(2,2,1); hold on;
scatter(rand_eig(logical(rand_eig_real),1),rand_eig(logical(rand_eig_real),2),10, 'b','filled');
scatter(real(rand_eig(logical(~rand_eig_real),1)),real(rand_eig(logical(~rand_eig_real),2)),10, 'r','filled');
axisminmax = xlim;
plot(linspace(axisminmax(1),axisminmax(2),100),-linspace(axisminmax(1),axisminmax(2),100),'k')
hline(0);vline(0);
xlabel('eig1')
ylabel('eig2')
legend('real', 'complex','y=-x')
title('Eigenvalues of random positive matrix, U(0,1)')

subplot(2,2,2); hold on;
scatter(dale_rand_eig(logical(dale_rand_eig_r),1),dale_rand_eig(logical(dale_rand_eig_r),2),10, 'b','filled');
scatter(real(dale_rand_eig(logical(~dale_rand_eig_r),1)),real(dale_rand_eig(logical(~dale_rand_eig_r),2)),10, 'r','filled');
axisminmax = xlim;
plot(linspace(axisminmax(1),axisminmax(2),100),linspace(axisminmax(1),axisminmax(2),100),'k')
hline(0);vline(0);
xlabel('eig1')
ylabel('eig2')
legend('real', 'complex','y=x')
title('Eigenvalues of random positive matrix + Dales law')

multiplier = 10;
for iter = 1:Nsamples
    A = multiplier*rand(2,2);
    rand_eig(iter,:) = eig(A);          % uniformly distributed random numbers in (0,1)
    dale_rand_eig(iter,:) = eig(A*B);   % Dales law applied.
    
    rand_eig_real(iter) = isreal(rand_eig(iter,:));
    dale_rand_eig_r(iter)  = isreal(dale_rand_eig(iter,:));
end

subplot(2,2,3); hold on;
scatter(rand_eig(logical(rand_eig_real),1),rand_eig(logical(rand_eig_real),2),10, 'b','filled');
scatter(real(rand_eig(logical(~rand_eig_real),1)),real(rand_eig(logical(~rand_eig_real),2)),10, 'r','filled');
axisminmax = xlim;
plot(linspace(axisminmax(1),axisminmax(2),100),-linspace(axisminmax(1),axisminmax(2),100),'k')
hline(0);vline(0);
xlabel('eig1')
ylabel('eig2')
legend('real', 'complex','y=-x')
title('Eigenvalues of random positive matrix U(0,10)')

subplot(2,2,4); hold on;
scatter(dale_rand_eig(logical(dale_rand_eig_r),1),dale_rand_eig(logical(dale_rand_eig_r),2),10, 'b','filled');
scatter(real(dale_rand_eig(logical(~dale_rand_eig_r),1)),real(dale_rand_eig(logical(~dale_rand_eig_r),2)),10, 'r','filled');
axisminmax = xlim;
plot(linspace(axisminmax(1),axisminmax(2),100),linspace(axisminmax(1),axisminmax(2),100),'k')
hline(0);vline(0);
xlabel('eig1')
ylabel('eig2')
legend('real', 'complex','y=x')
title('Eigenvalues of random positive matrix + Dales law')

savefig(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix.fig')
saveas(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix.png')
close(gcf)

%% Modifying self-connections
Nsamples = 10000;
rand_eig         = nan(Nsamples,2);
rand_eig_real    = [];
dale_rand_eig    = nan(Nsamples,2);
dale_rand_eig_r  = [];
alphas = [];
cmap = autumn(100);

multiplier = 1; % random matrix amplitude
for iter = 1:Nsamples
    alpha = rand();
    alphas(iter) = alpha; 
    A = multiplier*rand(2,2); % select a positive matrix
    rand_eig(iter,:) = eig((1-alpha)*eye(2,2) + alpha*A);          % uniformly distributed random numbers in (0,1)
    dale_rand_eig(iter,:) = eig((1-alpha)*eye(2,2) + alpha*A*B);   % Dales law applied.
    
    rand_eig_real(iter) = isreal(rand_eig(iter,:));
    dale_rand_eig_r(iter)  = isreal(dale_rand_eig(iter,:));
end

figure('Position', [1 41 1920 962]);

subplot(2,2,1); hold on;
scatter(rand_eig(logical(rand_eig_real),1),rand_eig(logical(rand_eig_real),2),...
    10,alphas(logical(rand_eig_real)),'x');
scatter(real(rand_eig(logical(~rand_eig_real),1)),real(rand_eig(logical(~rand_eig_real),2)),...
    10, alphas(logical(~rand_eig_real)),'o');
axisminmax = xlim;
plot(linspace(axisminmax(1),axisminmax(2),100),-linspace(axisminmax(1),axisminmax(2),100),'k')
c = colorbar; c.Label.String = 'alpha';
hline(0);vline(0);
xlabel('eig1')
ylabel('eig2')
legend('real', 'complex','y=-x')
title('Eigenvalues of random positive matrix U(0,1) with decay (1-\alpha)I')

subplot(2,2,2); hold on;
scatter(dale_rand_eig(logical(dale_rand_eig_r),1),dale_rand_eig(logical(dale_rand_eig_r),2),...
    10, alphas(logical(dale_rand_eig_r)),'x');
scatter(real(dale_rand_eig(logical(~dale_rand_eig_r),1)),real(dale_rand_eig(logical(~dale_rand_eig_r),2)),...
    10, alphas(logical(~dale_rand_eig_r)),'o');
axisminmax = xlim;
plot(linspace(axisminmax(1),axisminmax(2),100),linspace(axisminmax(1),axisminmax(2),100),'k')
c = colorbar; c.Label.String = 'alpha';
hline(0);vline(0);
xlabel('eig1')
ylabel('eig2')
l = legend('real', 'complex','y=x'); l.Location = 'northwest';
title('+ Dales law')

multiplier = 10; % random matrix amplitude
for iter = 1:Nsamples
    alpha = rand();
    alphas(iter) = alpha; 
    A = multiplier*rand(2,2); % select a positive matrix
    rand_eig(iter,:) = eig((1-alpha)*eye(2,2) + A);          % uniformly distributed random numbers in (0,1)
    dale_rand_eig(iter,:) = eig((1-alpha)*eye(2,2) + A*B);   % Dales law applied.
    
    rand_eig_real(iter) = isreal(rand_eig(iter,:));
    dale_rand_eig_r(iter)  = isreal(dale_rand_eig(iter,:));
end
subplot(2,2,3); hold on;
scatter(rand_eig(logical(rand_eig_real),1),rand_eig(logical(rand_eig_real),2),...
    10,alphas(logical(rand_eig_real)),'x');
scatter(real(rand_eig(logical(~rand_eig_real),1)),real(rand_eig(logical(~rand_eig_real),2)),...
    10, alphas(logical(~rand_eig_real)),'o');
axisminmax = xlim;
plot(linspace(axisminmax(1),axisminmax(2),100),-linspace(axisminmax(1),axisminmax(2),100),'k')
c = colorbar; c.Label.String = 'alpha';
hline(0);vline(0);
xlabel('eig1')
ylabel('eig2')
legend('real', 'complex','y=-x')
title('Eigenvalues of random positive matrix U(0,10) with decay (1-\alpha)I')

subplot(2,2,4); hold on;
scatter(dale_rand_eig(logical(dale_rand_eig_r),1),dale_rand_eig(logical(dale_rand_eig_r),2),...
    10, alphas(logical(dale_rand_eig_r)),'x');
scatter(real(dale_rand_eig(logical(~dale_rand_eig_r),1)),real(dale_rand_eig(logical(~dale_rand_eig_r),2)),...
    10, alphas(logical(~dale_rand_eig_r)),'o');
axisminmax = xlim;
plot(linspace(axisminmax(1),axisminmax(2),100),linspace(axisminmax(1),axisminmax(2),100),'k')
c = colorbar; c.Label.String = 'alpha';
hline(0);vline(0);
xlabel('eig1')
ylabel('eig2')
l = legend('real', 'complex','y=x'); l.Location = 'northwest';
title('+ Dales law')

savefig(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix_wDecay.fig')
saveas(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix_wDecay.png')
close(gcf)

%% Stability of the matrices

simulationFigure();

time = 1000;
multiplier = 1;

%{
while true
    alpha   = rand();
    A       = rand(2,2);
    
    [V,D] = eig(A*B);
    V
    D
    
end
%}
    


