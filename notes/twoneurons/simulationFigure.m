
function f = simulationFigure()
%% todo: adaptive figure size

% initialize variables
B = [1,0;0,-1]; % Dale's law matrix

W = rand(2,2);
m = 1;
alpha = rand();
NoiseStd = 1;
init = rand(2,1);
nsteps = 100;

%% GUI
guistruct = struct();

f = figure('Position', [1 41 1920 962]);
guistruct.f =f;
setappdata(f,'W', W);
setappdata(f,'m', m);
setappdata(f,'alpha', alpha);
setappdata(f,'noiseStd', NoiseStd);
setappdata(f,'init', init);

setappdata(f,'nsteps', nsteps);
setappdata(f,'B', B);

% controls for alpha
a_l = 150; a_b = 200; a_w = 300; a_h = 30;
guistruct.alphaControl = uicontrol('Parent',f,'Style','slider',...
    'SliderStep', [0.01, 0.1], ...
    'Position',[a_l a_b a_w a_h],...
    'value',alpha, 'min',0, 'max',1);
guistruct.alpha_label = annotation(f,'textbox','String',['alpha = ' sprintf('%4.2f',alpha)], ...
    'FontSize',10,'units','pix','Interpreter','Tex',...
    'Position', [a_l-100,a_b,90,a_h]);       
a_low = uicontrol('Parent',f,'Style','text','Position',[a_l-20, a_b-20,50,20],...
                'String','0','FontSize', 15);
a_high = uicontrol('Parent',f,'Style','text','Position',[a_l+a_w-20, a_b-20,50,20],...
                'String','1','FontSize', 15);

% controls for noise        
n_l = 150; n_b = 100; n_w = 300; n_h = 30;
maxnoise = 10;
guistruct.noiseControl = uicontrol('Parent',f,'Style','slider',...
    'SliderStep', [0.01, 0.1], ...
    'Position',[n_l n_b n_w n_h],...
    'value',alpha, 'min',0, 'max',maxnoise);
guistruct.n_label = annotation(f,'textbox','String',['Noise = ' sprintf('%4.2f',NoiseStd)], ...
    'FontSize',10,'units','pix','Interpreter','Tex',...
    'Position', [n_l-100,n_b,90,n_h]);       
n_low = uicontrol('Parent',f,'Style','text','Position',[n_l-20, n_b-20,50,20],...
                'String','0','FontSize', 15);
n_high = uicontrol('Parent',f,'Style','text','Position',[n_l+n_w-20, n_b-20,50,20],...
                'String',num2str(maxnoise),'FontSize', 15);
            
% controls for matrix noise
e_l = 150; e_b = 300; e_w = 300; e_h = 30;
maxEig = 10;
guistruct.EigControl = uicontrol('Parent',f,'Style','slider',...
    'Position',[e_l e_b e_w e_h],...
    'SliderStep', [0.1, 1], ...
    'value',m, 'min',0, 'max', maxEig);
guistruct.e_label = annotation(f,'textbox','String',['MaxEig = ' sprintf('%4.2f',m)], ...
    'FontSize',10,'units','pix','Interpreter','Tex',...
    'Position', [e_l-120,e_b,110,e_h]);       
e_low = uicontrol('Parent',f,'Style','text','Position',[e_l-20, e_b-20,50,20],...
                'String','0','FontSize', 15);
e_high = uicontrol('Parent',f,'Style','text','Position',[e_l+e_w-20, e_b-20,50,20],...
                'String',num2str(maxEig),'FontSize', 15);    

guistruct.ax1 = subplot(3,3,[2,3]); xlabel('Time (sec)');
guistruct.ax2 = subplot(3,3,[5,6]); xlabel('Time (sec)');
guistruct.ax3 = subplot(3,3,[8,9]); xlabel('Time (sec)');
linkaxes([ax1,ax2,ax3],'x');

b_l = 100; b_b = 880; b_w = 150; b_h = 75;
e = eig(W);
[V,D] = eig(W);
% e = [1+5i, 1-5i];

%% WRNN
annotation(f,'textbox','String','W=W_{RNN}', ...
    'FontSize',15,'units','pix','Interpreter','Tex',...
    'Position', [b_l-80,b_b-50,130,50]);       

str0 = sprintf('eigenvalues: %+4.2f %+4.2f i, %+4.2f %+4.2f i', ...
    real(e(1)),imag(e(1)), real(e(2)),imag(e(2)));
guistruct.wrnn.eig_label = annotation(f,'textbox','String',str0, ...
    'FontSize',15,'units','pix','Interpreter','Tex',...
    'Position', [b_l+50,b_b-50,400,50]);       

str1 = sprintf('Matrix: \n [%+4.2f, %+4.2f] \n [%+4.2f, %+4.2f]', ...
    W(1,1),W(1,2), W(2,1), W(2,2));
guistruct.wrnn.mat_label = annotation(f,'textbox','String',str1, ...
    'FontSize',20,'units','pix','Interpreter','Tex',...
    'Position', [b_l-50,b_b-200,200,150]);       

str2 = sprintf('ev1: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
    real(V(1,1)),imag(V(1,1)),  real(V(2,1)),imag(V(2,1)));
guistruct.wrnn.vec1_label = annotation(f,'textbox','String',str2, ...
    'FontSize',20,'units','pix','Interpreter','Tex',...
    'Position', [b_l+150,b_b-200,200,150]);       

str3 = sprintf('ev1: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
    real(V(1,2)),imag(V(1,2)),  real(V(2,2)),imag(V(2,2)));
guistruct.wrnn.vec2_label = annotation(f,'textbox','String',str3, ...
    'FontSize',20,'units','pix','Interpreter','Tex',...
    'Position', [b_l+350,b_b-200,200,150]);       

%% W
annotation(f,'textbox','String','(1-\alpha)I + \alphaW', ...
    'FontSize',15,'units','pix','Interpreter','Tex',...
    'Position', [b_l-80,b_b-50-250,130,50]);       

str0 = sprintf('eigenvalues: %+4.2f %+4.2f i, %+4.2f %+4.2f i', ...
    real(e(1)),imag(e(1)), real(e(2)),imag(e(2)));
guistruct.w.eig_label = annotation(f,'textbox','String',str0, ...
    'FontSize',15,'units','pix','Interpreter','Tex',...
    'Position', [b_l+50,b_b-50-250,400,50]);       

str1 = sprintf('Matrix: \n [%+4.2f, %+4.2f] \n [%+4.2f, %+4.2f]', ...
    W(1,1),W(1,2), W(2,1), W(2,2));
guistruct.w.mat_label = annotation(f,'textbox','String',str1, ...
    'FontSize',20,'units','pix','Interpreter','Tex',...
    'Position', [b_l-50,b_b-200-250,200,150]);       

str2 = sprintf('ev1: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
    real(V(1,1)),imag(V(1,1)),  real(V(2,1)),imag(V(2,1)));
guistruct.w.vec1_label = annotation(f,'textbox','String',str2, ...
    'FontSize',20,'units','pix','Interpreter','Tex',...
    'Position', [b_l+150,b_b-200-250,200,150]);       

str3 = sprintf('ev1: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
    real(V(1,2)),imag(V(1,2)),  real(V(2,2)),imag(V(2,2)));
guistruct.w.vec2_label = annotation(f,'textbox','String',str3, ...
    'FontSize',20,'units','pix','Interpreter','Tex',...
    'Position', [b_l+350,b_b-200-250,200,150]);       

%% Callbacks
guistruct.alphaControl.Callback = @(es,ed) changeData(guistruct,'alpha',es.Value);
guistruct.noiseControl.Callback = @(es,ed) changeData(guistruct,'noiseStd',es.Value);
guistruct.EigControl.Callback = @(es,ed) changeData(guistruct,'m',es.Value);

guistruct.NewMat = uicontrol(gcf,'Style', 'push', 'String', 'Generate New Matrix',...
    'Position', [b_l+50 b_b b_w b_h]);
guistruct.NewMat.Callback = @(eb,ed) generateNewWeights(guistruct);

guistruct.NewSim = uicontrol(gcf,'Style', 'push', 'String', 'Generate New Simulation',...
    'Position', [b_l+100+b_w, b_b, b_w, b_h]);
guistruct.NewSim.Callback = @(eb,ed) generateNewSim(guistruct);

generateNewWeights(guistruct); % initialize
end

function changeData(guistruct,varname,changeval)
    f = guistruct.f;
    
    setappdata(guistruct.f,varname,changeval);
    
    W           = getappdata(f,'W');
    m           = getappdata(f,'m');
    alpha       = getappdata(f,'alpha');
    NoiseStd    = getappdata(f,'noiseStd');
    
    guistruct.alpha_label.String = ['alpha = ' sprintf('%4.2f',alpha)];
    guistruct.n_label.String = ['Noise = ' sprintf('%4.2f',NoiseStd)];
    guistruct.e_label.String = ['MaxEig = ' sprintf('%4.2f',m)];
    
    updateWRNN(guistruct,m,W);    
    updateDecayedW(guistruct, m,W,alpha);
end

function generateNewWeights(guistruct)
    f = guistruct.f;
    W = getappdata(f,'W');
    m = getappdata(f,'m');
    alpha = getappdata(f,'alpha');                    
    B = getappdata(f,'B');
    
    A = rand(2,2); % select a positive matrix
    W = A*B; 
    
    setappdata(f,'W', W);
    
    updateWRNN(guistruct,m,W);    
    updateDecayedW(guistruct,m, W,alpha)
    generateNewSim(guistruct)
end

function updateWRNN(guistruct,m,W)
    %% WRNN
    W = m*W;
    e = eig(W);
    [V,D] = eig(W);
    
    if imag(e(2)) == 0
        str0 = sprintf('eigenvalues: %+4.2f, %+4.2f', ...
            real(e(1)), real(e(2)));    
    else
        str0 = sprintf('eigenvalues: %+4.2f %+4.2f i, %+4.2f %+4.2f i', ...
            real(e(1)),imag(e(1)), real(e(2)),imag(e(2)));
    end
    guistruct.wrnn.eig_label.String = str0;       

    str1 = sprintf('Matrix: \n [%+4.2f, %+4.2f] \n [%+4.2f, %+4.2f]', ...
        W(1,1),W(1,2), W(2,1), W(2,2));
    guistruct.wrnn.mat_label.String = str1;

    if imag(V(1,1)) == 0 
        str2 = sprintf('ev1: \n %+4.2f \n %+4.2f', ...
            real(V(1,1)), real(V(2,1)));
        str3 = sprintf('ev2: \n %+4.2f \n %+4.2f', ...
            real(V(1,2)), real(V(2,2)));
    else
        str2 = sprintf('ev1: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
            real(V(1,1)),imag(V(1,1)),  real(V(2,1)),imag(V(2,1)));
        str3 = sprintf('ev2: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
            real(V(1,2)),imag(V(1,2)),  real(V(2,2)),imag(V(2,2)));
    end
    guistruct.wrnn.vec1_label.String = str2;       
    guistruct.wrnn.vec2_label.String = str3;

end

function updateDecayedW(guistruct, m, W,alpha)
    %% W
    decay = (1-alpha)*eye(2,2);
    W = decay + alpha*m*W; % change W but do not save in appdata
    e = eig(W);
    [V,D] = eig(W);
    
    if imag(e(2)) == 0
        str0 = sprintf('eigenvalues: %+4.2f, %+4.2f', ...
            real(e(1)), real(e(2)));    
    else
        str0 = sprintf('eigenvalues: %+4.2f %+4.2f i, %+4.2f %+4.2f i', ...
            real(e(1)),imag(e(1)), real(e(2)),imag(e(2)));
    end
    guistruct.w.eig_label.String = str0;       

    str1 = sprintf('Matrix: \n [%+4.2f, %+4.2f] \n [%+4.2f, %+4.2f]', ...
        W(1,1),W(1,2), W(2,1), W(2,2));
    guistruct.w.mat_label.String = str1;

    if imag(V(1,1)) == 0 
        str2 = sprintf('ev1: \n %+4.2f \n %+4.2f', ...
            real(V(1,1)), real(V(2,1)));
        str3 = sprintf('ev2: \n %+4.2f \n %+4.2f', ...
            real(V(1,2)), real(V(2,2)));
    else
        str2 = sprintf('ev1: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
            real(V(1,1)),imag(V(1,1)),  real(V(2,1)),imag(V(2,1)));
        str3 = sprintf('ev2: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
            real(V(1,2)),imag(V(1,2)),  real(V(2,2)),imag(V(2,2)));
    end
    guistruct.w.vec1_label.String = str2;       
    guistruct.w.vec2_label.String = str3;
end

function generateNewSim(guistruct)
    f = guistruct.f;
    
    W           = getappdata(f,'W');
    m           = getappdata(f,'m');
    alpha       = getappdata(f,'alpha');
    NoiseStd    = getappdata(f,'noiseStd');
    init        = getappdata(f,'init');
    
    B           = getappdata(f,'B');
    nsteps      = getappdata(f,'nsteps');
    
    decay = (1-alpha)*eye(2,2);
    
    init = rand(1,2); % random (positive) initialization
    
    % run through recurrent matrix
    r(1,:) = init;
    r1(1,:) = init;
    r2(1,:) = init;
    for iter = 2:nsteps
        r(iter,:)  = decay*r(iter-1,:)' + alpha*m*W*r(iter-1,:)';
        r1(iter,:) = decay*r1(iter-1,:)' + alpha*relu(m*W*r1(iter-1,:)');
        r2(iter,:) = decay*r2(iter-1,:)' + alpha*relu(m*W*r2(iter-1,:)' + NoiseStd * sqrt(2/alpha)*rand(2,1)) ;
        %disp(r(iter,:))
        %disp(r1(iter,:))
        %disp(r2(iter,:))
    end
    
    axes(guistruct.ax1); cla; hold on;
    plot(r(:,1),'b'); plot(r(:,2),'r');
    title('Decay + dale'); 
    xlabel('Time'); legend('Exc', 'Inh')
    %set(gca,'Yscale','log')
    
    axes(guistruct.ax2); cla; hold on;
    plot(r1(:,1),'b'); plot(r1(:,2),'r');
    title('Decay + dale + relu'); 
    xlabel('Time'); legend('Exc', 'Inh')
    %set(gca,'Yscale','log')
    
    axes(guistruct.ax3); cla; hold on;
    plot(r2(:,1),'b'); plot(r2(:,2),'r');
    title('Decay + dale + relu + noise'); 
    xlabel('Time'); legend('Exc', 'Inh')
    %set(gca,'Yscale','log')
    
    setappdata(f,'init', init);
end

function x = relu(x)
    x(x<0) = 0;
end
