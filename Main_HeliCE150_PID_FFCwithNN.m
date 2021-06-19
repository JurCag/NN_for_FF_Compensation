% RAI - Semestrálna práca
% Juraj Cagáň, 200 349, 200349@vutbr.cz

%% Instructions
% Run this script by Sections.
% Don't run the entire script at once. 

% This script walks step by step through iterative process of extending measured data 
% used for training Neural Network, that is being used in feedforward compensation
% with PID feedback control to control the nonlinear dynamic model.

%% ___Heli Humusoft CE 150 - Est. Parameters___
%
%                (--------v--------)
%        __          __ __|__
%        (@)________(::(    |𖡄\__ 
%          `---------\___________)
%                     ___|___|___, 

% Main motor 
% Estimated quadratic equation coefficinets ->  Tau_1 = f(w_1)
% Tau_1 = a_1*w_1^2 + b_1*w_1
a_1 = 0.105;                    % [Nm/MU^2] 
b_1 = 0.00936;                  % [Nm/MU]

% Side motor 
% Estimated quadratic equation coefficinets ->  Tau_2 = f(w_2)
% Tau_2 = a_2*w_2^2 + b_2*w_2
a_2 = 0.033;                    % [Nm/MU^2]
b_2 = 0.0294;                   % [Nm/MU]

% Vertical plane -> Elevation
B_psi = 0.00184;                % [kg*m^2/s]    Coeff. of viscous damping
I = 0.00437;                    % [km*m^2]      Moment of inertia in vertical plane

% Horizontal plane -> Azimuth
B_phi = 0.00869;                % [kg*m^2/s]    Coeff. of viscous damping
I_phi = 0.00414;                % [km*m^2]      Moment of inertia in horizontal plane

% Speed of motors dependency on input voltage -> aproximated by 2nd order TF
% w_1 = 1/(T_1^2 - 1)^2 * u1
% w_2 = 1/(T_2^2 - 1)^2 * u2
T_1 = 0.3;                      % [s]           Time constant of the main motor
T_2 = 0.25;                     % [s]           Time constant of the side motor

% Reaction cross coupling -> aproximated by TF with gain, 1 zero and 1 pole
K_r = 0.00162;                  % [Nm/MU]       Gain constant  
T_0r = 2.7;                     % [s]           Zero time constant
T_pr = 0.75;                    % [s]           Pole time constant

K_Gyro = 0.015;                 % [Nm/s]        Gain of Gyroscopic moment
Tau_g = 0.0383;                 % [Nm]          Gravitation torque amplitude

%% Init. switch controls
dirIn_switch = 0;
FFC_switch = 0;
PID_u1_switch = 0;
PID_u2_switch = 0;
dirIn = 1;
desired_psi = 1;
FFC_u1_net = 1;
desired_phi = 1;
FFC_u2_net = 1;
T_u1 = 1;
T_u2 = 1;

%% Slow increase of input voltage u1
% To illustrate the behaviour of dynamic system
dirIn = 1;
dirIn_switch = 1;   % 0-OFF, 1-ON
fixPhi = 1;         % fix azimuth angle
sim_end = 100;

% Generate data
out = sim('Model_HeliCE150_PID_FFCwithNN','StopTime',num2str(sim_end));

% Load data
u1 = out.u1;
psi = out.psi;
t = out.tout;

%% Plot simulation data
figure()
hold on
set(gcf,'color','w');
subplot(2,1,1)
plot(t,u1, "LineWidth", 2)
title("Input voltage",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$u_1 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
ylim([0.45 0.6])
grid on

subplot(2,1,2)
hold on
plot([0 100], [45 45], "g--", "LineWidth", 1)	
plot([0 100], [90 90], "g--", "LineWidth", 1)
plot([0 100], [92 92], "r--", "LineWidth", 1)
plot([0 100], [135 135], "r--", "LineWidth", 1)
plot(t,psi, "b","LineWidth", 2)
title("Elevation",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
grid on

%% (1) First iteration -> partial elevation compensation (stable region)
% Slow increase of elevation angle psi
dirIn = 2;
dirIn_switch = 1;   % 0-OFF, 1-ON
fixPhi = 1;         % fix azimuth angle
sim_end = 100;

% Generate data
out = sim('Model_HeliCE150_PID_FFCwithNN','StopTime',num2str(sim_end));

% Load data
u1 = out.u1;
psi = out.psi;
t = out.tout;

%% (1) Plot measured data
figure()
hold on
set(gcf,'color','w');
subplot(3,1,1)
plot(t,u1, "LineWidth", 2)
title("Input voltage",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$u_1 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
ylim([0.45 0.6])
grid on

subplot(3,1,2)
hold on
plot([0 100], [45 45], "g--", "LineWidth", 1.5)
plot([0 100], [90 90], "g--", "LineWidth", 1.5)
plot(t,psi, "b","LineWidth", 2)
title("Elevation",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
grid on

subplot(3,1,3)
plot(psi, u1, 'LineWidth',2);
xlim([45 90])
ylim([0.46 0.58])
grid on
title("Measured data: $u_1 = f(\psi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$u_1 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);

%% (1) Prepare data for training NN
num_train = 0.8;
inputs = psi';
targets = u1';
[inputs_train,targets_train,inputs_test,targets_test] = prepTrainTestData(inputs,targets,num_train,1);

%% (1) Train NN
hiddenCnt = 5;
trainingFcn = 'trainlm';
net = feedforwardnet(hiddenCnt, trainingFcn);

net.trainParam.epochs = 1000;
net.trainParam.lr = 0.05;

net = configure(net, inputs_train, targets_train);
net = train(net, inputs_train, targets_train); 

% Prediction
targets_pred = sim(net, inputs_test);

% -> Skip training by loading already trained net
% net_e1 = load('TrainedNets/ffnet_5_elev_only_1');
% net = net_e1.net;

%% (1) Plot prediction
figure()
set(gcf,'color','w');
hold on
plot(inputs_test, targets_test, 'bo');
plot(inputs_test, targets_pred, 'r.');
legend('Test data','Predicted values','Interpreter','latex','fontweight','bold','fontsize',15)
title("NN Prediction: $u_1 = f(\psi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$u_{1} [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\psi [^{\circ}]$','Interpreter','latex','fontweight','bold','fontsize',15)
grid on

%% (1) Feedforward control of elevation in open loop
dirIn_switch = 0;
FFC_u1 = 1;
FFC_switch = 1;
desired_psi = 1;
FFC_u1_net = 1;

sim_end = 100;

% Generate data
out = sim('Model_HeliCE150_PID_FFCwithNN','StopTime',num2str(sim_end));

% Load data
psi = out.psi;
psi_d = out.psi_d;
phi = out.phi;

%% (1) Plot feedforward control of elevation in open loop
figure()
hold on
set(gcf,'color','w');
plot(t,psi,'b', "LineWidth", 2)
plot(t,psi_d,'k', "LineWidth", 1)
plot([0 100], [45 45], "g--", "LineWidth", 1)
plot([0 100], [135 135], "r--", "LineWidth", 1)
title("Elevation",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
legend('$\psi$ ','$\psi_{d}$','Interpreter','latex','fontweight','bold','fontsize',15)
grid on

%% (2) Second iteration -> azimuth and elevation compensation (stable region)
% Load prepared datasets - 3 measurements
D_a0   = load('MeasuredDatasets/Data100_ea_a0.mat');      % Data azimuth 0° angle
D_aP90 = load('MeasuredDatasets/Data100_ea_aP90.mat');    % Data azimuth Positive 90° angle
D_aN90 = load('MeasuredDatasets/Data100_ea_aN90.mat');    % Data azimuth Negative 90° angle
 
U1  = [D_aP90.u1'  , D_a0.u1' , D_aN90.u1' ];
U2  = [D_aP90.u2'  , D_a0.u2' , D_aN90.u2' ];
PSI = [D_aP90.psi' , D_a0.psi', D_aN90.psi'];
PHI = [D_aP90.phi' , D_a0.phi', D_aN90.phi'];

%% (2) Visualize measured datasets and highlight stable regions

% u1 = f(psi,phi)
figure()
set(gcf,'color','w');
subplot(1,2,1)
hold on
plot3(PSI(1:length(D_a0.psi)), PHI(1:length(D_a0.psi)),U1(1:length(D_a0.psi)),'Linewidth',2)
plot3(PSI(length(D_a0.psi)+1:2*length(D_a0.psi)), PHI(length(D_a0.psi)+1:2*length(D_a0.psi)),U1(length(D_a0.psi)+1:2*length(D_a0.psi)),'Linewidth',2)
plot3(PSI(2*length(D_a0.psi)+1:3*length(D_a0.psi)), PHI(2*length(D_a0.psi)+1:3*length(D_a0.psi)),U1(2*length(D_a0.psi)+1:3*length(D_a0.psi)),'Linewidth',2)
plot3([45 45],[-130 130],[0.46 0.46],'g--')
plot3([45 45],[-130 -130],[0.46 0.58],'g--')
plot3([90 90],[-130 130],[0.46 0.46],'g--')
plot3([90 90],[-130 -130],[0.46 0.58],'g--')
plot3([45 90],[130 130],[0.46 0.46],'g--')
plot3([90 90],[130 130],[0.46 0.58],'g--')
plot3([45 90],[-130 -130],[0.46 0.46],'g--')
text(90,90,0.57,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,0,0.57,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-90,0.57,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("Measured data: $u_1 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_1 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlim([40 95])
ylim([-150 150])
zlim([0.46 0.58])
grid on
view(-124,15)

% u2 = f(psi,phi)
set(gcf,'color','w');
subplot(1,2,2)
hold on
plot3(PSI(1:length(D_a0.psi)), PHI(1:length(D_a0.psi)),U2(1:length(D_a0.psi)),'Linewidth',2)
plot3(PSI(length(D_a0.psi)+1:2*length(D_a0.psi)), PHI(length(D_a0.psi)+1:2*length(D_a0.psi)),U2(length(D_a0.psi)+1:2*length(D_a0.psi)),'Linewidth',2)
plot3(PSI(2*length(D_a0.psi)+1:3*length(D_a0.psi)), PHI(2*length(D_a0.psi)+1:3*length(D_a0.psi)),U2(2*length(D_a0.psi)+1:3*length(D_a0.psi)),'Linewidth',2)
plot3([45 45],[-130 130],[0.024 0.024],'g--')
plot3([45 45],[-130 -130],[0.024 0.032],'g--')
plot3([90 90],[-130 130],[0.024 0.024],'g--')
plot3([90 90],[-130 -130],[0.024 0.032],'g--')
plot3([45 90],[130 130],[0.024 0.024],'g--')
plot3([90 90],[130 130],[0.024 0.032],'g--')
plot3([45 90],[-130 -130],[0.024 0.024],'g--')
text(90,95,0.0305,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,5,0.0305,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-85,0.0305,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("Measured data: $u_2 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_2 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlim([40 95])
ylim([-150 150])
zlim([0.024 0.032])
grid on
view(-124,15)

%% (2) Prepare data for training NN
num_train = 0.8;
inputs = [PSI;
          PHI]; % inputs[NxM] N-inputs, M-measured points 
targets = [U1;
           U2]; % targets[PxM] P-targets, M-measured points
[inputs_train,targets_train,inputs_test,targets_test] = prepTrainTestData(inputs,targets,num_train,1);

%% (2) Train NN -> 2 inputs, 2 outputs (ffnet [10,5] hidden neurons)
% hiddenCnt = [10 5];
% 
% trainingFcn = 'trainlm';
% net = feedforwardnet(hiddenCnt, trainingFcn);
% 
% net.trainParam.epochs = 1000;
% net.trainParam.lr = 0.05;
% net.trainParam.min_grad = 1e-6;
% 
% net = configure(net, inputs_train, targets_train);
% 
% net = train(net, inputs_train, targets_train);

% -> Skip training by loading already trained net
net_10_5_ea2 = load('TrainedNets/ffnet_10_5_elev_azim_2');
net = net_10_5_ea2.net;

%% (2) Prediction inside stable region (ffnet [10,5] hidden neurons)
psi_mat = linspace(45,90,40);
phi_mat = linspace(-130,130,40);

psimesh = meshgrid(psi_mat,phi_mat)';
phimesh = meshgrid(phi_mat,psi_mat);

my_inputs = zeros(2,numel(psimesh));
for i = 1:numel(psimesh)
    my_inputs(1,i) = psimesh(i);
    my_inputs(2,i) = phimesh(i);
end

% SIMULATE OUTPUTS
targets_pred = sim(net,my_inputs);

% First output u1 = f(psi,phi)
targets_pred_u1 = targets_pred(1,:);

% Rearrange to plot surface
targetsSurf = zeros(length(phi_mat));
for i = 1:length(phi_mat)
    targetsSurf(i,:) = targets_pred_u1((i-1)*length(phi_mat)+1:i*length(phi_mat));
end

figure()
subplot(1,2,1)
set(gcf,'color','w');
hold on 
[xx,ff] = meshgrid(psi_mat,phi_mat);
plot3(inputs_train(1,:),inputs_train(2,:),targets_train(1,:),"ro",'LineWidth',0.5,'MarkerFaceColor', 'r')
plot3(inputs_test(1,:),inputs_test(2,:),targets_test(1,:),"bo",'LineWidth',0.5,'MarkerFaceColor', 'b')
s_u1 = surf(xx,ff,targetsSurf,'FaceAlpha',0.4); % approximated surface
text(90,90,0.575,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,0,0.575,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-90,0.575,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("NN Prediction: $u_1 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_1 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
grid on
view(-124,24)


% Second output u2 = f(psi,phi)
targets_pred_u2 = targets_pred(2,:);

% Rearrange to plot surface
targetsSurf = zeros(length(phi_mat));
for i = 1:length(phi_mat)
    targetsSurf(i,:) = targets_pred_u2((i-1)*length(phi_mat)+1:i*length(phi_mat));
end

subplot(1,2,2)
set(gcf,'color','w');
hold on 
[xx,ff] = meshgrid(psi_mat,phi_mat);
plot3(inputs_train(1,:),inputs_train(2,:),targets_train(2,:),"ro",'LineWidth',0.5,'MarkerFaceColor', 'r')
plot3(inputs_test(1,:),inputs_test(2,:),targets_test(2,:),"bo",'LineWidth',0.5,'MarkerFaceColor', 'b')
s_u2 = surf(xx,ff,targetsSurf,'FaceAlpha',0.4); % approximated surface
text(90,90,0.0305,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,0,0.0305,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-90,0.0305,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("NN Prediction: $u_2 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_2 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
legend('Train data','Test data','Prediction','Interpreter','latex','fontweight','bold','fontsize',12);
grid on
view(-124,24)

%% (2) Train NN -> 2 inputs, 2 outputs (ffnet [2,2] hidden neurons)
hiddenCnt = [2 2];

trainingFcn = 'trainlm';
net = feedforwardnet(hiddenCnt, trainingFcn);

net.trainParam.epochs = 1000;
net.trainParam.lr = 0.05;
net.trainParam.min_grad = 1e-7;

net = configure(net, inputs_train, targets_train);

net = train(net, inputs_train, targets_train);

% -> Skip training by loading already trained net
% net_2_2_ea2 = load('TrainedNets/ffnet_2_2_elev_azim_2');
% net = net_2_2_ea2.net;

%% (2) Prediction inside stable region (ffnet [2,2] hidden neurons)
% ffnet [2,2] gives better results inside the whole region
psi_mat = linspace(45,90,40);
phi_mat = linspace(-130,130,40);

psimesh = meshgrid(psi_mat,phi_mat)';
phimesh = meshgrid(phi_mat,psi_mat);

my_inputs = zeros(2,numel(psimesh));
for i = 1:numel(psimesh)
    my_inputs(1,i) = psimesh(i);
    my_inputs(2,i) = phimesh(i);
end

% SIMULATE OUTPUTS
targets_pred = sim(net,my_inputs);

% First output u1 = f(psi,phi)
targets_pred_u1 = targets_pred(1,:);

% Rearrange to plot surface
targetsSurf = zeros(length(phi_mat));
for i = 1:length(phi_mat)
    targetsSurf(i,:) = targets_pred_u1((i-1)*length(phi_mat)+1:i*length(phi_mat));
end

figure()
subplot(1,2,1)
set(gcf,'color','w');
hold on 
[xx,ff] = meshgrid(psi_mat,phi_mat);
plot3(inputs_train(1,:),inputs_train(2,:),targets_train(1,:),"ro",'LineWidth',0.5,'MarkerFaceColor', 'r')
plot3(inputs_test(1,:),inputs_test(2,:),targets_test(1,:),"bo",'LineWidth',0.5,'MarkerFaceColor', 'b')
s_u1 = surf(xx,ff,targetsSurf,'FaceAlpha',0.4); % approximated surface
text(90,90,0.575,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,0,0.575,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-90,0.575,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("NN Prediction: $u_1 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_1 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
grid on
view(-124,24)


% Second output u2 = f(psi,phi)
targets_pred_u2 = targets_pred(2,:);

% Rearrange to plot surface
targetsSurf = zeros(length(phi_mat));
for i = 1:length(phi_mat)
    targetsSurf(i,:) = targets_pred_u2((i-1)*length(phi_mat)+1:i*length(phi_mat));
end

subplot(1,2,2)
set(gcf,'color','w');
hold on 
[xx,ff] = meshgrid(psi_mat,phi_mat);
plot3(inputs_train(1,:),inputs_train(2,:),targets_train(2,:),"ro",'LineWidth',0.5,'MarkerFaceColor', 'r')
plot3(inputs_test(1,:),inputs_test(2,:),targets_test(2,:),"bo",'LineWidth',0.5,'MarkerFaceColor', 'b')
s_u2 = surf(xx,ff,targetsSurf,'FaceAlpha',0.4); % approximated surface
text(90,90,0.0305,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,0,0.0305,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-90,0.0305,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("NN Prediction: $u_2 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_2 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
legend('Train data','Test data','Prediction','Interpreter','latex','fontweight','bold','fontsize',12);
grid on
view(-124,24)

%% (2) PID control with FFC compensation - Elevation and Azimuth (stable region)
desired_phi = 1;
FFC_u2_net = 1;
PID_u1_switch = 1;
PID_u2_switch = 1;
dirIn_switch = 0;
FFC_u1_net = 2;

fixPhi = 0;         % free azimuth angle
sim_end = 100;

% Generate data
out = sim('Model_HeliCE150_PID_FFCwithNN','StopTime',num2str(sim_end));

% Load data
psi = out.psi;
psi_d = out.psi_d;
phi = out.phi;
phi_d = out.phi_d;
t = out.tout;

%% (2) Plot PID control with FFC compensation - Elevation and Azimuth (stable region)
figure()
set(gcf,'color','w');
subplot(2,1,1)
hold on
plot(t,psi,'b', "LineWidth", 2)
plot(t,psi_d,'k', "LineWidth", 1)
plot([0 100], [45 45], "g--", "LineWidth", 1)
plot([0 100], [135 135], "r--", "LineWidth", 1)
title("Elevation",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
legend('$\psi$ ','$\psi_{d}$','Interpreter','latex','fontweight','bold','fontsize',15)
grid on

subplot(2,1,2)
hold on
plot(t,phi,'Color', [1 0 227/255], "LineWidth", 2)
plot(t,phi_d,'k', "LineWidth", 1)
plot([0 100], [-130 -130], "g--", "LineWidth", 1)	
plot([0 100], [130 130], "g--", "LineWidth", 1)
title("Azimuth",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
legend('$\phi$ ','$\phi_{d}$','Interpreter','latex','fontweight','bold','fontsize',15)
grid on

%% (3) Third iteration -> azimuth and elevation compensation (unstable region)
% Load prepared datasets - 3 measurements
D2500_aN90 = load('MeasuredDatasets/Data2500_aN90_eOver90_filt.mat');      
D2500_a0   = load('MeasuredDatasets/Data2500_a0_eOver90_filt.mat');    
D2500_aP90 = load('MeasuredDatasets/Data2500_aP90_eOver90_filt.mat'); 

U1  = [D2500_aP90.u1'  , D2500_a0.u1' , D2500_aN90.u1' ];
U2  = [D2500_aP90.u2'  , D2500_a0.u2' , D2500_aN90.u2' ];
PSI = [D2500_aP90.psi' , D2500_a0.psi', D2500_aN90.psi'];
PHI = [D2500_aP90.phi' , D2500_a0.phi', D2500_aN90.phi'];

%% (3) Visualize measured datasets and highlight unstable regions
% u1 = f(psi,phi)
figure()
set(gcf,'color','w');
subplot(1,2,1)
hold on
plot3(PSI(1:length(D2500_aN90.psi)), PHI(1:length(D2500_aN90.psi)),U1(1:length(D2500_aN90.psi)),'Linewidth',2,"Color",[0.4660 0.6740 0.1880])
plot3(PSI(length(D2500_aN90.psi)+1:2*length(D2500_aN90.psi)), PHI(length(D2500_aN90.psi)+1:2*length(D2500_aN90.psi)),U1(length(D2500_aN90.psi)+1:2*length(D2500_aN90.psi)),'Linewidth',2,"Color",[0.3010 0.7450 0.9330])
plot3(PSI(2*length(D2500_aN90.psi)+1:3*length(D2500_aN90.psi)), PHI(2*length(D2500_aN90.psi)+1:3*length(D2500_aN90.psi)),U1(2*length(D2500_aN90.psi)+1:3*length(D2500_aN90.psi)),'Linewidth',2,"Color",[0.4940 0.1840 0.5560])
plot3([90 90],[-130 130],[0.520 0.520],'r--')
plot3([90 90],[-130 -130],[0.520 0.565],'r--')
plot3([135 135],[-130 130],[0.520 0.520],'r--')
plot3([135 135],[-130 -130],[0.520 0.565],'r--')
plot3([135 135],[128 128],[0.520 0.565],'r--')
plot3([85 135],[130 130],[0.520 0.520],'g--')
plot3([85 135],[-130 -130],[0.520 0.520],'g--')
plot3([135 135],[130 130],[0.520 0.565],'g--')
plot3([134 134],[-130 -130],[0.520 0.565],'g--')
plot3([89 89],[-130 -130],[0.520 0.565],'g--')
text(120,90,0.55,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(120,0,0.55,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(120,-90,0.55,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("Measured data: $u_1 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_1 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlim([85 140])
ylim([-150 150])
zlim([0.520 0.565])
grid on
view(-124,15)

% u2 = f(psi,phi)
set(gcf,'color','w');
subplot(1,2,2)
hold on
plot3(PSI(1:length(D2500_aN90.psi)), PHI(1:length(D2500_aN90.psi)),U2(1:length(D2500_aN90.psi)),'Linewidth',1,"Color",[0.4660 0.6740 0.1880])
plot3(PSI(length(D2500_aN90.psi)+1:2*length(D2500_aN90.psi)), PHI(length(D2500_aN90.psi)+1:2*length(D2500_aN90.psi)),U2(length(D2500_aN90.psi)+1:2*length(D2500_aN90.psi)),'Linewidth',1,"Color",[0.3010 0.7450 0.9330])
plot3(PSI(2*length(D2500_aN90.psi)+1:3*length(D2500_aN90.psi)), PHI(2*length(D2500_aN90.psi)+1:3*length(D2500_aN90.psi)),U2(2*length(D2500_aN90.psi)+1:3*length(D2500_aN90.psi)),'Linewidth',1,"Color",[0.4940 0.1840 0.5560])
plot3([90 90],[-130 130],[0.026 0.026],'r--')
plot3([90 90],[-130 -130],[0.026 0.03],'r--')
plot3([135 135],[-130 130],[0.026 0.026],'r--')
plot3([135 135],[-130 -130],[0.026 0.03],'r--')
plot3([135 135],[128 128],[0.026 0.03],'r--')
plot3([85 135],[130 130],[0.026 0.026],'g--')
plot3([85 134],[-130 -130],[0.026 0.026],'g--')
plot3([135 135],[130 130],[0.026 0.03],'g--')
plot3([134 134],[-130 -130],[0.026 0.03],'g--')
plot3([89 89],[-130 -130],[0.026 0.03],'g--')
text(125,110,0.0295,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(125,10,0.0295,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(125,-85,0.0295,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("Measured data: $u_2 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_2 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlim([85 140])
ylim([-150 150])
zlim([0.026 0.03])
grid on
view(-124,15)

%% (3) Prepare data for training NN
num_train = 0.8;
inputs = [PSI;
          PHI]; % inputs[NxM] N-inputs, M-measured points 
targets = [U1;
           U2]; % targets[PxM] P-targets, M-measured points
[inputs_train,targets_train,inputs_test,targets_test] = prepTrainTestData(inputs,targets,num_train,1);

%% (3) Train NN -> 2 inputs, 2 outputs (ffnet [2,1,2] hidden neurons)
% hiddenCnt = [2 1 2];
% 
% trainingFcn = 'trainlm';
% net = feedforwardnet(hiddenCnt, trainingFcn);
% 
% net.trainParam.epochs = 1000;
% net.trainParam.lr = 0.05;
% net.trainParam.min_grad = 1e-7;
% 
% net = configure(net, inputs_train, targets_train);
% 
% net = train(net, inputs_train, targets_train);

% -> Skip training by loading already trained net
net_2_1_2_ea3 = load('TrainedNets/ffnet_2_1_2_elev_azim_3');
net = net_2_1_2_ea3.net;

%% (3) Prediction inside unstable region (ffnet [2,1,2] hidden neurons)
psi_mat = linspace(90,115,40);
phi_mat = linspace(-130,130,40);

psimesh = meshgrid(psi_mat,phi_mat)';
phimesh = meshgrid(phi_mat,psi_mat);

my_inputs = zeros(2,numel(psimesh));
for i = 1:numel(psimesh)
    my_inputs(1,i) = psimesh(i);
    my_inputs(2,i) = phimesh(i);
end

% SIMULATE OUTPUTS
targets_pred = sim(net,my_inputs);

% First output u1 = f(psi,phi)
targets_pred_u1 = targets_pred(1,:);

% Rearrange to plot surface
targetsSurf = zeros(length(phi_mat));
for i = 1:length(phi_mat)
    targetsSurf(i,:) = targets_pred_u1((i-1)*length(phi_mat)+1:i*length(phi_mat));
end

figure()
subplot(1,2,1)
set(gcf,'color','w');
hold on 
[xx,ff] = meshgrid(psi_mat,phi_mat);
plot3(inputs_train(1,:),inputs_train(2,:),targets_train(1,:),"ro",'LineWidth',0.5,'MarkerFaceColor', 'r')
plot3(inputs_test(1,:),inputs_test(2,:),targets_test(1,:),"bo",'LineWidth',0.5,'MarkerFaceColor', 'b')
s_u1 = surf(xx,ff,targetsSurf,'FaceAlpha',0.4); % approximated surface
text(90,90,0.565,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,0,0.565,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-90,0.565,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("NN Prediction: $u_1 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_1 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
grid on
xlim([85 140])
ylim([-150 150])
zlim([0.520 0.565])
view(-153,7)


% Second output u2 = f(psi,phi)
targets_pred_u2 = targets_pred(2,:);

% Rearrange to plot surface
targetsSurf = zeros(length(phi_mat));
for i = 1:length(phi_mat)
    targetsSurf(i,:) = targets_pred_u2((i-1)*length(phi_mat)+1:i*length(phi_mat));
end

subplot(1,2,2)
set(gcf,'color','w');
hold on 
[xx,ff] = meshgrid(psi_mat,phi_mat);
plot3(inputs_train(1,:),inputs_train(2,:),targets_train(2,:),"ro",'LineWidth',0.5,'MarkerFaceColor', 'r')
plot3(inputs_test(1,:),inputs_test(2,:),targets_test(2,:),"bo",'LineWidth',0.5,'MarkerFaceColor', 'b')
s_u2 = surf(xx,ff,targetsSurf,'FaceAlpha',0.4); % approximated surface
text(90,90,0.0304,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,0,0.0304,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-90,0.0304,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("NN Prediction: $u_2 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_2 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
legend('Train data','Test data','Prediction','Interpreter','latex','fontweight','bold','fontsize',12);
xlim([85 140])
ylim([-150 150])
zlim([0.026 0.03])
grid on
view(-153,7)


%% (3) PID control with FFC compensation - Elevation and Azimuth (unstable region)
desired_psi = 2;
desired_phi = 2;
PID_u1_switch = 1;
PID_u2_switch = 1;
dirIn_switch = 0;
FFC_u1_net = 3;
FFC_u2_net = 2;
T_u1 = 0;

fixPhi = 0;         % free azimuth angle
sim_end = 200;

% Generate data
out = sim('Model_HeliCE150_PID_FFCwithNN','StopTime',num2str(sim_end));

% Load data
psi = out.psi;
psi_d = out.psi_d;
phi = out.phi;
phi_d = out.phi_d;
t = out.tout;

%% (3) Plot PID control with FFC compensation - Elevation and Azimuth (unstable region)
figure()
set(gcf,'color','w');
subplot(2,1,1)
hold on
plot(t,psi,'b', "LineWidth", 1.5)
plot(t,psi_d,'k', "LineWidth", 1)
plot([0 sim_end], [45 45], "g--", "LineWidth", 1)	
plot([0 sim_end], [89.5 89.5], "g--", "LineWidth", 1)
plot([0 sim_end], [91 91], "r--", "LineWidth", 1)
plot([0 sim_end], [135 135], "r--", "LineWidth", 1)
title("Elevation",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
legend('$\psi$ ','$\psi_{d}$','Interpreter','latex','fontweight','bold','fontsize',15)
grid on

subplot(2,1,2)
hold on
plot(t,phi,'Color', [1 0 227/255], "LineWidth", 1.5)
plot(t,phi_d,'k', "LineWidth", 1)
title("Azimuth",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
plot([0 sim_end], [-130 -130], "g--", "LineWidth", 1)	
plot([0 sim_end], [130 130], "g--", "LineWidth", 1)
legend('$\phi$ ','$\phi_{d}$','Interpreter','latex','fontweight','bold','fontsize',15)
ylim([-150 150])
grid on

%% (4) Combination of Training data from stable and unstable region of psi

% STABLE
D_a0   = load('MeasuredDatasets/Data100_ea_a0.mat');                        % Data azimuth 0° angle
D_aP90 = load('MeasuredDatasets/Data100_ea_aP90.mat');                      % Data azimuth Positive 90° angle
D_aN90 = load('MeasuredDatasets/Data100_ea_aN90.mat');                      % Data azimuth Negative 90° angle

% UNSTABLE
D2500_aN90   = load('MeasuredDatasets/Data2500_aN90_eOver90_filt.mat');     % Data azimuth Negative 90° angle   
D2500_a0 = load('MeasuredDatasets/Data2500_a0_eOver90_filt.mat');           % Data azimuth 0° angle
D2500_aP90 = load('MeasuredDatasets/Data2500_aP90_eOver90_filt.mat');       % Data azimuth Positive 90° angle

U1  = [D_aP90.u1'  , D_a0.u1' , D_aN90.u1' , D2500_aP90.u1'  , D2500_a0.u1' , D2500_aN90.u1' ];
U2  = [D_aP90.u2'  , D_a0.u2' , D_aN90.u2' , D2500_aP90.u2'  , D2500_a0.u2' , D2500_aN90.u2' ];
PSI = [D_aP90.psi' , D_a0.psi', D_aN90.psi', D2500_aP90.psi' , D2500_a0.psi', D2500_aN90.psi'];
PHI = [D_aP90.phi' , D_a0.phi', D_aN90.phi', D2500_aP90.phi' , D2500_a0.phi', D2500_aN90.phi'];

%% (4) Visualize measured datasets and highlight stable/unstable regions

% u1 = f(psi,phi)
figure()
set(gcf,'color','w');
subplot(1,2,1)
hold on
plot3(D_aP90.psi,D_aP90.phi,D_aP90.u1,'Linewidth',1.5)
plot3(D_a0.psi,D_a0.phi,D_a0.u1,'Linewidth',1.5)
plot3(D_aN90.psi,D_aN90.phi,D_aN90.u1,'Linewidth',1.5)
plot3(D2500_aP90.psi,D2500_aP90.phi,D2500_aP90.u1,'Linewidth',1.5,"Color",[0.4660 0.6740 0.1880])
plot3(D2500_a0.psi,D2500_a0.phi,D2500_a0.u1,'Linewidth',1.5,"Color",[0.3010 0.7450 0.9330])
plot3(D2500_aN90.psi,D2500_aN90.phi,D2500_aN90.u1,'Linewidth',1.5,"Color",[0.4940 0.1840 0.5560])
plot3([45 45],[-130 130],[0.46 0.46],'g--')
plot3([45 45],[-130 -130],[0.46 0.58],'g--')
plot3([90 90],[-130 130],[0.46 0.46],'g--')
plot3([90 90],[-130 -130],[0.46 0.58],'g--')
plot3([45 90],[130 130],[0.46 0.46],'g--')
plot3([90 90],[130 130],[0.46 0.58],'g--')
plot3([45 90],[-130 -130],[0.46 0.46],'g--')
plot3([92 92],[-130 130],[0.46 0.46],'r--')
plot3([92 92],[-130 -130],[0.46 0.58],'r--')
plot3([135 135],[-130 130],[0.46 0.46],'r--')
plot3([135 135],[-130 -130],[0.46 0.58],'r--')
plot3([135 135],[128 128],[0.46 0.58],'r--')
plot3([92 92],[128 128],[0.46 0.58],'r--')
plot3([85 135],[130 130],[0.46 0.46],'g--')
plot3([85 135],[-130 -130],[0.46 0.46],'g--')
plot3([135 135],[130 130],[0.46 0.58],'g--')
plot3([134 134],[-130 -130],[0.46 0.58],'g--')
text(90,110,0.568,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,20,0.568,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-70,0.568,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("Measured data: $u_1 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_1 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlim([40 140])
ylim([-150 150])
zlim([0.46 0.58])
grid on
view(-124,15)

% u2 = f(psi,phi)
set(gcf,'color','w');
subplot(1,2,2)
hold on
plot3(D_aP90.psi,D_aP90.phi,D_aP90.u2,'Linewidth',1.5)
plot3(D_a0.psi,D_a0.phi,D_a0.u2,'Linewidth',1.5)
plot3(D_aN90.psi,D_aN90.phi,D_aN90.u2,'Linewidth',1.5)
plot3(D2500_aP90.psi,D2500_aP90.phi,D2500_aP90.u2,'Linewidth',1.5,"Color",[0.4660 0.6740 0.1880])
plot3(D2500_a0.psi,D2500_a0.phi,D2500_a0.u2,'Linewidth',1.5,"Color",[0.3010 0.7450 0.9330])
plot3(D2500_aN90.psi,D2500_aN90.phi,D2500_aN90.u2,'Linewidth',1.5,"Color",[0.4940 0.1840 0.5560])
plot3([45 45],[-130 130],[0.024 0.024],'g--')
plot3([45 45],[-130 -130],[0.024 0.03],'g--')
plot3([90 90],[-130 130],[0.024 0.024],'g--')
plot3([90 90],[-130 -130],[0.024 0.03],'g--')
plot3([45 90],[130 130],[0.024 0.024],'g--')
plot3([90 90],[130 130],[0.024 0.03],'g--')
plot3([45 90],[-130 -130],[0.024 0.024],'g--')
plot3([92 92],[-130 130],[0.024 0.024],'r--')
plot3([92 92],[-130 -130],[0.024 0.03],'r--')
plot3([135 135],[-130 130],[0.024 0.024],'r--')
plot3([135 135],[-130 -130],[0.024 0.03],'r--')
plot3([135 135],[128 128],[0.024 0.03],'r--')
plot3([92 92],[128 128],[0.024 0.03],'r--')
plot3([85 135],[130 130],[0.024 0.024],'g--')
plot3([85 135],[-130 -130],[0.024 0.024],'g--')
plot3([135 135],[130 130],[0.024 0.03],'g--')
plot3([134 134],[-130 -130],[0.024 0.03],'g--')
text(130,125,0.0297,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(130,25,0.0297,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(130,-70,0.0297,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("Measured data: $u_2 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_2 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlim([40 140])
ylim([-150 150])
zlim([0.024 0.03])
grid on
view(-124,15)

%% (4) Prepare data for training NN
num_train = 0.8;
inputs = [PSI;
          PHI]; % inputs[NxM] N-inputs, M-measured points 
targets = [U1;
           U2]; % targets[PxM] P-targets, M-measured points
[inputs_train,targets_train,inputs_test,targets_test] = prepTrainTestData(inputs,targets,num_train,1);

%% (4) Train NN
% hiddenCnt = [2 2];
% 
% trainingFcn = 'trainlm';
% net = feedforwardnet(hiddenCnt, trainingFcn);
% 
% net.trainParam.epochs = 1000;
% net.trainParam.lr = 0.05;
% net.trainParam.min_grad = 1e-7;
% 
% net = configure(net, inputs_train, targets_train);
% 
% net = train(net, inputs_train, targets_train);

% -> Skip training by loading already trained net
net_2_2_ea4combined = load('TrainedNets/ffnet_2_2_elev_azim_4combined');
net = net_2_2_ea4combined.net;

%% (4) Prediction inside stable and unstable region
psi_mat = linspace(45,115,40);
phi_mat = linspace(-130,130,40);

psimesh = meshgrid(psi_mat,phi_mat)';
phimesh = meshgrid(phi_mat,psi_mat);

my_inputs = zeros(2,numel(psimesh));
for i = 1:numel(psimesh)
    my_inputs(1,i) = psimesh(i);
    my_inputs(2,i) = phimesh(i);
end

% SIMULATE OUTPUTS
targets_pred = sim(net,my_inputs);

% First output u1 = f(psi,phi)
targets_pred_u1 = targets_pred(1,:);

% Rearrange to plot surface
targetsSurf = zeros(length(phi_mat));
for i = 1:length(phi_mat)
    targetsSurf(i,:) = targets_pred_u1((i-1)*length(phi_mat)+1:i*length(phi_mat));
end

figure()
subplot(1,2,1)
set(gcf,'color','w');
hold on 
[xx,ff] = meshgrid(psi_mat,phi_mat);
plot3(inputs_train(1,:),inputs_train(2,:),targets_train(1,:),"ro",'LineWidth',0.5,'MarkerFaceColor', 'r')
plot3(inputs_test(1,:),inputs_test(2,:),targets_test(1,:),"bo",'LineWidth',0.5,'MarkerFaceColor', 'b')
s_u1 = surf(xx,ff,targetsSurf,'FaceAlpha',0.4); % approximated surface
text(90,90,0.575,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,0,0.575,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-90,0.575,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("NN Prediction: $u_1 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_1 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
grid on
xlim([40 140])
ylim([-150 150])
zlim([0.46 0.58])
view(-124,24)


% Second output u2 = f(psi,phi)
targets_pred_u2 = targets_pred(2,:);

% Rearrange to plot surface
targetsSurf = zeros(length(phi_mat));
for i = 1:length(phi_mat)
    targetsSurf(i,:) = targets_pred_u2((i-1)*length(phi_mat)+1:i*length(phi_mat));
end

subplot(1,2,2)
set(gcf,'color','w');
hold on 
[xx,ff] = meshgrid(psi_mat,phi_mat);
plot3(inputs_train(1,:),inputs_train(2,:),targets_train(2,:),"ro",'LineWidth',0.5,'MarkerFaceColor', 'r')
plot3(inputs_test(1,:),inputs_test(2,:),targets_test(2,:),"bo",'LineWidth',0.5,'MarkerFaceColor', 'b')
s_u2 = surf(xx,ff,targetsSurf,'FaceAlpha',0.4); % approximated surface
text(90,90,0.0305,'$\phi =90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,0,0.0305,'$\phi =0^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
text(90,-90,0.0305,'$\phi =-90^{\circ}$ ','Interpreter','latex','fontweight','bold','fontsize',12);
title("NN Prediction: $u_2 = f(\psi,\phi)$",'Interpreter','latex','fontweight','bold','fontsize',15)
xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
zlabel('$u_2 [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
legend('Train data','Test data','Prediction','Interpreter','latex','fontweight','bold','fontsize',12);
grid on
xlim([40 140])
ylim([-150 150])
zlim([0.024 0.03])
view(-124,24)


%% (4) PID control with FFC compensation - Elevation and Azimuth (stable and unstable region) 
% Combined Dataset -> One NN for psi<45°,110°>
desired_psi = 2;
desired_phi = 2;
PID_u1_switch = 1;
PID_u2_switch = 1;
dirIn_switch = 0;
FFC_u1_net = 4;
FFC_u2_net = 3;
T_u1 = 0;

fixPhi = 0;         % free azimuth angle
sim_end = 200;

% Generate data
out = sim('Model_HeliCE150_PID_FFCwithNN','StopTime',num2str(sim_end));

% Load data
psi = out.psi;
psi_d = out.psi_d;
phi = out.phi;
phi_d = out.phi_d;
t = out.tout;

%% (4) Plot PID control with FFC compensation - Elevation and Azimuth (stable and unstable region)
% Combined Dataset -> One NN for psi<45°,110°>
figure()
set(gcf,'color','w');
subplot(2,1,1)
hold on
plot(t,psi,'b', "LineWidth", 1.5)
plot(t,psi_d,'k', "LineWidth", 1)
plot([0 sim_end], [45 45], "g--", "LineWidth", 1)	
plot([0 sim_end], [89.5 89.5], "g--", "LineWidth", 1)
plot([0 sim_end], [91 91], "r--", "LineWidth", 1)
plot([0 sim_end], [135 135], "r--", "LineWidth", 1)
title("Elevation",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
legend('$\psi$ ','$\psi_{d}$','Interpreter','latex','fontweight','bold','fontsize',15)
grid on

subplot(2,1,2)
hold on
plot(t,phi,'Color', [1 0 227/255], "LineWidth", 1.5)
plot(t,phi_d,'k', "LineWidth", 1)
title("Azimuth",'Interpreter','latex','fontweight','bold','fontsize',15)
ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
xlabel('$\boldmath{t [s]}$','Interpreter','latex','fontweight','bold','fontsize',15)
plot([0 sim_end], [-130 -130], "g--", "LineWidth", 1)	
plot([0 sim_end], [130 130], "g--", "LineWidth", 1)
legend('$\phi$ ','$\phi_{d}$','Interpreter','latex','fontweight','bold','fontsize',15)
ylim([-150 150])
grid on






































