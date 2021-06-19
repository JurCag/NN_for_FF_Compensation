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
% w_1 = 1/(T_1^2 - 1)^2
% w_2 = 1/(T_2^2 - 1)^2
T_1 = 0.3;                      % [s]           Time constant of the main motor
T_2 = 0.25;                     % [s]           Time constant of the side motor

% Reaction cross coupling -> aproximated by TF with gain, 1 zero and 1 pole
K_r = 0.00162;                  % [Nm/MU]       Gain constant  
T_0r = 2.7;                     % [s]           Zero time constant
T_pr = 0.75;                    % [s]           Pole time constant

K_Gyro = 0.015;                 % [Nm/s]        Gain of Gyroscopic moment
Tau_g = 0.0383;                 % [Nm]          Gravitation torque amplitude

T_u1 = 0;                       % [s]           Time constant of LP filter for feedfoward compensation of main motor voltage u1
T_u2 = 1;                       % [s]           Time constant of LP filter for feedfoward compensation of side motor voltage u2