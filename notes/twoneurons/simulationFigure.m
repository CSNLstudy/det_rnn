
function f = simulationFigure()
%% todo: adaptive figure size

% initialize variables
B = [1,0;0,-1]; % Dale's law matrix

W = rand(2,2);
m = 1;
alpha = rand();
NoiseStd = 1;
init = rand(2,1);
nsteps = 1000;

%% GUI
guistruct = struct();

pos_absolute= [1 41 1920 962]; % todo: https://www.mathworks.com/matlabcentral/answers/215444-set-uicontrol-position-relative-to-a-subplot
w = 1920; h = 962;
pos_2normal = [w h w h];

f = figure('Units', 'normal', 'Position', [0 0 0.95 0.95]);
guistruct.f =f;
setappdata(f,'W', W);
setappdata(f,'m', m);
setappdata(f,'alpha', alpha);
setappdata(f,'noiseStd', NoiseStd);
setappdata(f,'init', init);

setappdata(f,'nsteps', nsteps);
setappdata(f,'B', B);

% controls for alpha
a_l = 150/w; a_b = 200/h; a_w = 300/w; a_h = 30/h;
guistruct.alphaControl = uicontrol('Parent',f,'Style','slider',...
    'SliderStep', [0.01, 0.1], ...
    'Units', 'normal','Position',[a_l a_b a_w a_h],...
    'value',alpha, 'min',0, 'max',1);
guistruct.alpha_label = annotation(f,'textbox','String',['alpha = ' sprintf('%4.2f',alpha)], ...
    'FontSize',10,'units','normal','Interpreter','Tex',...
    'Units', 'normal','Position', [a_l-120/w,a_b,110/h,a_h]);       
a_low = uicontrol('Parent',f,'Style','text',...
    'Units', 'normal','Position',[a_l-20/w, a_b-20/h,50/w,20/h],...
    'String','0','FontSize', 15);
a_high = uicontrol('Parent',f,'Style','text',...
    'Units', 'normal','Position',[a_l+a_w-20/w, a_b-20/h,50/w,20/h],...
    'String','1','FontSize', 15);

% controls for noise        
n_l = 150/w; n_b = 100/h; n_w = 300/w; n_h = 30/h;
maxnoise = 10;
guistruct.noiseControl = uicontrol('Parent',f,'Style','slider',...
    'Units', 'normal','Position',[n_l n_b n_w n_h],...
    'value',alpha, 'min',0, 'max',maxnoise,...
    'SliderStep', [0.001, 0.02]);
guistruct.n_label = annotation(f,'textbox','String',['Noise = ' sprintf('%4.2f',NoiseStd)], ...
    'FontSize',10,'units','normal','Interpreter','Tex',...
    'Units', 'normal','Position', [n_l-120/w,n_b,110/w,n_h]);       
n_low = uicontrol('Parent',f,'Style','text',...
    'Units', 'normal','Position',[n_l-20/w, n_b-20/h,50/w,20/h],...
                'String','0','FontSize', 15);
n_high = uicontrol('Parent',f,'Style','text',...
    'Units', 'normal','Position',[n_l+n_w-20/w, n_b-20/h,50/w,20/h],...
                'String',num2str(maxnoise),'FontSize', 15);
            
% controls for matrix noise
e_l = 150/w; e_b = 300/h; e_w = 300/w; e_h = 30/h;
maxEig = 10;
guistruct.EigControl = uicontrol('Parent',f,'Style','slider',...
    'Units', 'normal','Position',[e_l e_b e_w e_h],...
    'value',m, 'min',0, 'max', maxEig,...
    'SliderStep', [0.001,0.02]);
guistruct.e_label = annotation(f,'textbox','String',['MaxEig = ' sprintf('%4.2f',m)], ...
    'FontSize',10,'units','normal','Interpreter','Tex',...
    'Units', 'normal','Position', [e_l-120/w,e_b,110/w,e_h]);       
e_low = uicontrol('Parent',f,'Style','text',...
    'Units', 'normal','Position',[e_l-20/w, e_b-20/h,50/w,20/h],...
                'String','0','FontSize', 15);
e_high = uicontrol('Parent',f,'Style','text',...
    'Units', 'normal','Position',[e_l+e_w-20/w, e_b-20/h,50/w,20/h],...
                'String',num2str(maxEig),'FontSize', 15);    

% [1 41 1920 962]
t_l = 1750/w; t_b = 92/h; t_w = 100/w;  t_h = 50/h;
guistruct.toglogscale.ax3 = uicontrol('Parent',f,'Style','togglebutton','String', 'log scale',...
                        'Units', 'normal','Position',[t_l, (t_b+50/h), t_w, t_h]);
guistruct.toglogscale.ax2 = uicontrol('Parent',f,'Style','togglebutton','String', 'log scale',...
    'Units', 'normal','Position',[t_l, (t_b + 900/h*1/3 + 50/h), t_w, t_h]);
guistruct.toglogscale.ax1 = uicontrol('Parent',f,'Style','togglebutton','String', 'log scale',...
    'Units', 'normal','Position',[t_l, (t_b + 900/h*2/3 +50/h), t_w, t_h]);
guistruct.togautoscale.ax3 = uicontrol('Parent',f,'Style','pushbutton','String', 'auto scale',...
    'Units', 'normal','Position',[t_l, (t_b +100/h), t_w, t_h]);
guistruct.togautoscale.ax2 = uicontrol('Parent',f,'Style','pushbutton','String', 'auto scale',...
    'Units', 'normal','Position',[t_l, (t_b + 900/h*1/3 + 100/h), t_w, t_h]);
guistruct.togautoscale.ax1 = uicontrol('Parent',f,'Style','pushbutton','String', 'auto scale',...
    'Units', 'normal','Position',[t_l, (t_b + 900/h*2/3 +100/h), t_w, t_h]);
guistruct.togTime = uicontrol('Parent',f,'Style','togglebutton','String', 'full time',...
    'Units', 'normal','Position',[t_l, (t_b + 900/h*2/3 +180/h), t_w, t_h]);
                    
guistruct.ax1 = subplot(3,3,[2,3]); xlabel('Time (sec)');
guistruct.ax2 = subplot(3,3,[5,6]); xlabel('Time (sec)');
guistruct.ax3 = subplot(3,3,[8,9]); xlabel('Time (sec)');

b_l = 100/w; b_b = 880/h; b_w = 150/w; b_h = 75/h;
e = eig(W);
[V,D] = eig(W);
% e = [1+5i, 1-5i];

%% WRNN
annotation(f,'textbox','String','W=W_{RNN}', ...
    'FontSize',15,'Interpreter','Tex',...
    'Units', 'normal','Position', [b_l-80/w,b_b-50/h,130/w,50/h]);       

str0 = sprintf('eigenvalues: %+4.2f %+4.2f i, %+4.2f %+4.2f i', ...
    real(e(1)),imag(e(1)), real(e(2)),imag(e(2)));
guistruct.wrnn.eig_label = annotation(f,'textbox','String',str0, ...
    'FontSize',15,'units','normal','Interpreter','latex',...
    'Position', [b_l+50/w,b_b-50/h,400/w,50/h]);       

str1 = sprintf('Matrix: \n [%+4.2f, %+4.2f] \n [%+4.2f, %+4.2f]', ...
    W(1,1),W(1,2), W(2,1), W(2,2));
guistruct.wrnn.mat_label = annotation(f,'textbox','String',str1, ...
    'FontSize',20,'units','normal','Interpreter','Tex',...
    'Position', [b_l-50/w,b_b-200/h,200/w,150/h]);       

str2 = sprintf('ev1: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
    real(V(1,1)),imag(V(1,1)),  real(V(2,1)),imag(V(2,1)));
guistruct.wrnn.vec1_label = annotation(f,'textbox','String',str2, ...
    'FontSize',20,'units','normal','Interpreter','Tex',...
    'Position', [b_l+150/w,b_b-200/h,200/w,150/h]);       

str3 = sprintf('ev1: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
    real(V(1,2)),imag(V(1,2)),  real(V(2,2)),imag(V(2,2)));
guistruct.wrnn.vec2_label = annotation(f,'textbox','String',str3, ...
    'FontSize',20,'units','normal','Interpreter','Tex',...
    'Position', [b_l+350/w,b_b-200/h,200/w,150/h]);       

%% W
annotation(f,'textbox','String','(1-\alpha)I + \alphaW', ...
    'FontSize',15,'units','normal','Interpreter','Tex',...
    'Position', [b_l-80/w,b_b-50/h-250/h,130/w,50/h]);       

str0 = sprintf('eigenvalues: %+4.2f %+4.2f i, %+4.2f %+4.2f i', ...
    real(e(1)),imag(e(1)), real(e(2)),imag(e(2)));
guistruct.w.eig_label = annotation(f,'textbox','String',str0, ...
    'FontSize',15,'units','normal','Interpreter','latex',...
    'Position', [b_l+50/w,b_b-50/h-250/h,400/w,50/h]);       

str1 = sprintf('Matrix: \n [%+4.2f, %+4.2f] \n [%+4.2f, %+4.2f]', ...
    W(1,1),W(1,2), W(2,1), W(2,2));
guistruct.w.mat_label = annotation(f,'textbox','String',str1, ...
    'FontSize',20,'units','normal','Interpreter','latex',...
    'Position', [b_l-50/w,b_b-200/h-250/h,200/h,150/h]);       

str2 = sprintf('ev1: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
    real(V(1,1)),imag(V(1,1)),  real(V(2,1)),imag(V(2,1)));
guistruct.w.vec1_label = annotation(f,'textbox','String',str2, ...
    'FontSize',20,'units','normal','Interpreter','Tex',...
    'Position', [b_l+150/w,b_b-200/h-250/h,200/w,150/h]);       

str3 = sprintf('ev1: \n %+4.2f %+4.2f i \n %+4.2f %+4.2f i', ...
    real(V(1,2)),imag(V(1,2)),  real(V(2,2)),imag(V(2,2)));
guistruct.w.vec2_label = annotation(f,'textbox','String',str3, ...
    'FontSize',20,'units','normal','Interpreter','Tex',...
    'Position', [b_l+350/w,b_b-200/h-250/h,200/w,150/h]);       

%% Callbacks
guistruct.alphaControl.Callback = @(es,ed) changeData(guistruct,'alpha',es.Value);
guistruct.noiseControl.Callback = @(es,ed) changeData(guistruct,'noiseStd',es.Value);
guistruct.EigControl.Callback = @(es,ed) changeData(guistruct,'m',es.Value);

guistruct.NewMat = uicontrol(gcf,'Style', 'push', 'String', 'Generate New Matrix',...
    'units','normal','Position', [b_l+50/w b_b b_w b_h]);
guistruct.NewMat.Callback = @(eb,ed) generateNewWeights(guistruct);

guistruct.NewSim = uicontrol(gcf,'Style', 'push', 'String', 'Generate New Simulation',...
    'units','normal','Position', [b_l+100/w+b_w, b_b, b_w, b_h]);
guistruct.NewSim.Callback = @(eb,ed) generateNewSim(guistruct);

guistruct.toglogscale.ax1.Callback = @(eb,ed) togglescale(guistruct, 1, 1);
guistruct.toglogscale.ax2.Callback = @(eb,ed) togglescale(guistruct, 1, 2);
guistruct.toglogscale.ax3.Callback = @(eb,ed) togglescale(guistruct, 1, 3);
guistruct.togautoscale.ax1.Callback = @(eb,ed) togglescale(guistruct, 2, 1);
guistruct.togautoscale.ax2.Callback = @(eb,ed) togglescale(guistruct, 2, 2);
guistruct.togautoscale.ax3.Callback = @(eb,ed) togglescale(guistruct, 2, 3);

guistruct.togTime.Callback = @(eb,ed) toggleTime(guistruct);

generateNewWeights(guistruct); % initialize
end

function togglescale(guistruct, type, ax)

    if ax == 1
        currax = guistruct.ax1;
        if type == 1
            currbutt = guistruct.toglogscale.ax1;
        elseif type ==2
            currbutt = guistruct.togautoscale.ax1; 
        end
    elseif ax ==2
        currax = guistruct.ax2;
        if type == 1
            currbutt = guistruct.toglogscale.ax2;
        elseif type ==2
            currbutt = guistruct.togautoscale.ax2; 
        end
    elseif ax == 3
        currax = guistruct.ax3;
        if type == 1
            currbutt = guistruct.toglogscale.ax3;
        elseif type ==2
            currbutt = guistruct.togautoscale.ax3; 
        end
    end
    
    axes(currax); 
    
    if type == 1
        if currbutt.Value == currbutt.Max
            set(gca,'Yscale','log')        % change to log scale
        else
            set(gca,'Yscale','linear')        % change to log scale
        end
    elseif type == 2
        axis 'auto y'
    end
end

function toggleTime(guistruct)
    linkaxes([guistruct.ax1,guistruct.ax2,guistruct.ax3],'x');
    axes(guistruct.ax1)
    Nsteps    = getappdata(guistruct.f,'nsteps');

    if guistruct.togTime.Value == guistruct.togTime.Max
        xlim([0,Nsteps])        
    else
        xlim([0,Nsteps/10])    
    end
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
        str0 = sprintf('eigenvalues: r=%+4.2f, angle =%+4.2f deg', ...
            abs(e(1)),angle(e(1))/pi*180);
    end
    guistruct.wrnn.eig_label.String = str0;       

    str1 = sprintf('Matrix: \n [%+4.2f, %+4.2f] \n [%+4.2f, %+4.2f]', ...
        W(1,1),W(1,2), W(2,1), W(2,2));
    guistruct.wrnn.mat_label.String = str1;

    if (imag(V(1,1)) == 0) && (imag(V(2,1)) == 0) 
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
        str0 = sprintf('eigenvalues: r=%+4.2f, angle=%+4.2f deg', ...
            abs(e(1)),angle(e(1))/pi*180);
    end
    guistruct.w.eig_label.String = str0;       

    str1 = sprintf('Matrix: \n [%+4.2f, %+4.2f] \n [%+4.2f, %+4.2f]', ...
        W(1,1),W(1,2), W(2,1), W(2,2));
    guistruct.w.mat_label.String = str1;

    if (imag(V(1,1)) == 0) && (imag(V(2,1)) == 0) 
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
    xlabel('Time'); legend('Exc', 'Inh');
    %set(gca,'Yscale','log')
    
    axes(guistruct.ax2); cla; hold on;
    plot(r1(:,1),'b'); plot(r1(:,2),'r');
    title('Decay + dale + relu'); 
    xlabel('Time'); legend('Exc', 'Inh');
    %set(gca,'Yscale','log')
    
    axes(guistruct.ax3); cla; hold on;
    plot(r2(:,1),'b'); plot(r2(:,2),'r');
    title('Decay + dale + relu + noise'); 
    xlabel('Time'); legend('Exc', 'Inh');
    %set(gca,'Yscale','log')
    
    linkaxes([guistruct.ax1,guistruct.ax2,guistruct.ax3],'x');
    xlim([0,1000])
    
    setappdata(f,'init', init);
end

function x = relu(x)
    x(x<0) = 0;
end
