Nsamples = 10000;
B = [1,0;0,-1]; % Dale's law matrix

gamma = true;
if gamma == true
    pd = makedist('Gamma','a',0.1,'b',1); 
end

%% Eigenvalues of positive matrices and Dale's law.
rand_eig         = nan(Nsamples,2);
rand_eig_real    = [];
dale_rand_eig    = nan(Nsamples,2);
dale_rand_eig_r  = [];

for iter = 1:Nsamples
    if gamma == true
        A = pd.random(2,2);
    else
        A = rand(2,2);
    end
    
    rand_eig(iter,:) = eig(A);          % uniformly distributed random numbers in (0,1)
    dale_rand_eig(iter,:) = eig(A*B);   % Dales law applied.
    
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
if gamma == false
    title('Eigenvalues of random positive matrix U(0,1)')
else
    title('Eigenvalues of random positive matrix Gamma(0.1,1)')
end

subplot(2,2,2); hold on;
scatter(dale_rand_eig(logical(dale_rand_eig_r),1),dale_rand_eig(logical(dale_rand_eig_r),2),10, 'b','filled');
scatter(real(dale_rand_eig(logical(~dale_rand_eig_r),1)),real(dale_rand_eig(logical(~dale_rand_eig_r),2)),10, 'r','filled');
axisminmax = xlim;
plot(linspace(axisminmax(1),axisminmax(2),100),linspace(axisminmax(1),axisminmax(2),100),'k')
hline(0);vline(0);
xlabel('eig1')
ylabel('eig2')
legend('real', 'complex','y=x')
title('+ Dales law  ')

multiplier = 10;
for iter = 1:Nsamples
    if gamma == true
        A = pd.random(2,2);
    else
        A = rand(2,2);
    end
    
    rand_eig(iter,:) = eig(multiplier*A);          % uniformly distributed random numbers in (0,1)
    dale_rand_eig(iter,:) = eig(multiplier*A*B);   % Dales law applied.
    
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
if gamma == false
    title('Eigenvalues of random positive matrix U(0,10)')
else
    title('Eigenvalues of random positive matrix 10*Gamma(0.1,1)')
end

subplot(2,2,4); hold on;
scatter(dale_rand_eig(logical(dale_rand_eig_r),1),dale_rand_eig(logical(dale_rand_eig_r),2),10, 'b','filled');
scatter(real(dale_rand_eig(logical(~dale_rand_eig_r),1)),real(dale_rand_eig(logical(~dale_rand_eig_r),2)),10, 'r','filled');
axisminmax = xlim;
plot(linspace(axisminmax(1),axisminmax(2),100),linspace(axisminmax(1),axisminmax(2),100),'k')
hline(0);vline(0);
xlabel('eig1')
ylabel('eig2')
legend('real', 'complex','y=x')
title(' + Dales law')

if gamma == true
    savefig(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix_gamma.fig')
    saveas(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix_gamma.png')
else
    savefig(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix.fig')
    saveas(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix.png')
end

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
    
    if gamma == true
        A = pd.random(2,2);
    else
        A = rand(2,2);
    end    
    rand_eig(iter,:) = eig((1-alpha)*eye(2,2) + alpha*multiplier*A);          % uniformly distributed random numbers in (0,1)
    dale_rand_eig(iter,:) = eig((1-alpha)*eye(2,2) + alpha*multiplier*A*B);   % Dales law applied.
    
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

if gamma == false
    title('Eigenvalues of random positive matrix U(0,1) with decay (1-\alpha)I')
else
    title('Eigenvalues of random positive matrix Gamma(0.1,1) with decay (1-\alpha)I')
end

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
    
    if gamma == true
        A = pd.random(2,2);
    else
        A = rand(2,2);
    end
    rand_eig(iter,:) = eig((1-alpha)*eye(2,2) + alpha*multiplier*A);          % uniformly distributed random numbers in (0,1)
    dale_rand_eig(iter,:) = eig((1-alpha)*eye(2,2) + alpha*multiplier*A*B);   % Dales law applied.
    
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

if gamma == false
    title('Eigenvalues of random positive matrix U(0,10) with decay (1-\alpha)I')
else
    title('Eigenvalues of random positive matrix 10*Gamma(0.1,1) with decay (1-\alpha)I')
end

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

if gamma == true
    savefig(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix_wDecay_gamma.fig')
    saveas(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix_wDecay_gamma.png')
else
    savefig(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix_wDecay.fig')
    saveas(gcf,'D:\proj\det_rnn\notes\twoneurons\positiveMatrix_wDecay.png')
end

close(gcf)

%% Stability of the matrices

% simulationFigure();
% lemma1: if you "interpolate" with an identity matrix, the eigenvectors
% stay the same => obvious. use definition of eigenvalue.
alpha = rand();
A = eye(2); %[1,0;0,-1];%eye(2); diag(rand(2,1));
W = 0.5-rand(2,2); %rand(2,2); %0.5-rand(2,2); %rand(2,2);

[V1, D1] = eig(W);
[V2, D2] = eig(A);
[V3, D3] = eig((1-alpha)*A + alpha*W);
disp(V1);disp(V2);disp(V3);

[U,S,V] = svd(W);% m*W = U,S,V
