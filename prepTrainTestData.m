function [inputs_train,targets_train,inputs_test,targets_test] = prepTrainTestData(inputs,targets,num_train,plotData)
% prepTrainTestData - prepares Training and Test data for feedforward neural network
% inputs[NxM]   N-number of inputs, M-measured points
% targets[PxM]  P-number of targets, M-measured points


% Initialize training data
inputs_train = zeros(size(inputs,1), round(num_train*size(inputs,2)));   
targets_train = zeros(size(targets,1), round(num_train*size(targets,2)));  

% Initialize test data
inputs_test = zeros(size(inputs,1), round((1-num_train)*size(inputs,2)));  
targets_test = zeros(size(targets,1), round((1-num_train)*size(targets,2))); 


perm = randperm(size(inputs,2));

for i = 1:length(perm)
    
    if i < round(num_train*size(inputs,2)+1) % Add to training data
        
        for k = 1:size(inputs,1)
            inputs_train(k,i) = inputs(k,perm(i));
        end
        
        for n = 1:size(targets,1)
            targets_train(n,i) = targets(n,perm(i));
        end
        
    elseif i > round(num_train*size(inputs,2)) % Add to test data
      
        for k = 1:size(inputs,1)
            inputs_test(k,i-round(num_train*size(inputs,2))) = inputs(k,perm(i));
        end
        
        for n = 1:size(targets,1)
            targets_test(n,i-round(num_train*size(inputs,2))) = targets(n,perm(i));
        end
        
    end
end


if plotData == 1 && size(inputs,1)==1 && size(targets,1)==1
    figure()
    set(gcf,'color','w');
    hold on
    plot(inputs_train, targets_train, 'go')
    plot(inputs_test, targets_test, 'b.')
    legend('Training dataset','Test dataset','Interpreter','latex','fontweight','bold','fontsize',15)
    grid on
end


if (plotData == 1 && size(inputs,1)==2 && size(targets,1)<=2)
    for i = 1:size(targets,1)
        figure()
        set(gcf,'color','w');
        hold on
        grid on
        plot3(inputs_train(1,:),inputs_train(2,:),targets_train(i,:),"go",'LineWidth',0.5,'MarkerFaceColor', 'g')
        plot3(inputs_test(1,:),inputs_test(2,:),targets_test(i,:),"bo",'LineWidth',0.5,'MarkerFaceColor', 'b')
        legend('Training data','Test data')
        title(['u_',num2str(i)],'fontweight','bold','fontsize',15)
        xlabel('$\psi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
        ylabel('$\phi [^{\circ}]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
        zlabel('$u [V]$ ','Interpreter','latex','fontweight','bold','fontsize',15);
        grid on
        view(-160,20)
    end
end






end






