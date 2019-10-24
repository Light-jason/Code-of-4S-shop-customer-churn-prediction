% BP�㷨ѵ�����(�����)��֪��
% ��MLP��ģ�Ͳ���������Ȩ��weights, ƫ����biase
% ��MLP���ڶ��������񣬹���������logistic�����⣬�������ļ����ҲΪlogistic
function [weights,biase,probs_of_positve,error_rate_of_TestSet,final_loss,time_consume] = MLP(x_train,y_train,MLP_struc,x_test,y_test,initial_para,opts)
%% �Ż�������ģ�ͳ�ʼ������ȱʡ����
tic
rng(2018)
if(nargin < 3);error('���������������3����');end
num_of_input = MLP_struc(1); num_of_output = MLP_struc(end);
if(size(x_train,2) ~= num_of_input || size(y_train,2) ~= num_of_output)
    error('�����������(��)����Ԫ������������(��ǩ��)��ͬ��')
end
len = length(MLP_struc);
if(len>5);warning('�����϶࣬ѵ��ʱ�䳤!�����������ʱ�䣬���������ĵȴ���׼��');end
if(nargin == 5)   
    % �����Ż�������ȱʡ����
    opts.epoch = 3e3;
    opts.learning_rate = 0.05;
    opts.batch_size = ceil(size(x_train,1)/40);
    opts.momentum = 0.9;   
    % ����ѵ��Ŀ�꣺lossС�������ֵʱ����ֹͣ����
    opts.training_object = 1e-4;
    % ��ʼ����ȱʡ����
    coef = 0.1;
    for i = 1 : len-1
       initial_para.weights{i} = coef*randn(MLP_struc(i),MLP_struc(i+1));
       initial_para.biase{i} = zeros(1,MLP_struc(i+1));
    end
  
end

%% ����MLP�ĳ�ʼѵ������
weights = initial_para.weights; biase = initial_para.biase; 

    %��ʼ����������Ϊ0; ��ʼ�ݶ�����Ϊ0��
v_weights = cell(1,len-1); v_biase = cell(1,len-1); 
d_weights = cell(1,len-1); d_biase = cell(1,len-1);
for i = 1 : len-1
    v_weights{i} = zeros(size(initial_para.weights{i}));
    v_biase{i} = zeros(size(initial_para.biase{i}));
    d_weights{i} = zeros(size(initial_para.weights{i}));
    d_biase{i} = zeros(size(initial_para.biase{i}));
end

L = cell(1,len); % ���ڼ�¼MLP��������ֵ
delta = cell(1,len-1); % ���ڼ�¼���򴫲������ֵ
ce_loss = zeros(opts.epoch,1); % ���ڼ�¼�����صı仯
lr = opts.learning_rate; momentum = opts.momentum;

%% ���弤�����logistic����
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));

%% ʹ��С�����Ķ����ݶ��½���ѵ�� 
    % �ȷֺ�batch
num_of_sample = size(x_train,1);
remainder = mod(num_of_sample, opts.batch_size);
start = 1 : opts.batch_size : (num_of_sample-remainder);
final = opts.batch_size : opts.batch_size : (num_of_sample-remainder);
final(end) = final(end) + remainder;
batch_index = [start;final]';
    % ��ʼѵ��
for i = 0 : opts.epoch-1
    j = mod(i,size(batch_index,1)) + 1 ; %ѡbatch
    
    % forward propagationǰ�򴫲���������������L1~Ln����n��
    L{1} = x_train(batch_index(j,1):batch_index(j,2),:);
    for k = 2 : len
        L{k} = sigm(L{k-1},weights{k-1},biase{k-1});
    end
    
    % back propagation ���򴫲�������"�ݶ�"delta3, delta2, delta1
    target = y_train(batch_index(j,1):batch_index(j,2),:); % ��ʵ���
    %dE_wrt_Ln = -target./L{len} + (1-target)./(1-L{len}); % ��������ʧ������������ƫ����,���ʽ���ǲ��ȶ��ģ����ܻ����NaN; 
    % dE_wrt_Ln = L{len} - target; % ���Ƕ�����ʧ������������ƫ���������ڻع�����
        % ���򴫲�
    % delta{len-1} = dE_wrt_Ln .* L{len} .* (1-L{len});
    delta{len-1} = L{len} - target; % ����д�Ͳ������NaN��
    for k = (len-2):-1:1  
        delta{k} = delta{k+1} * weights{k+1}' .* L{k+1} .* (1-L{k+1});
    end
    
    % �����ݶ�,�����²���
    n = size(L{1},1);
    for k = (len-1):-1:1  % Ϊ�ˡ�BP���ķ��򱣳�һ�£�д��len:1,��д��1:lenҲ�ǶԵ�
        % �����ݶ�
        d_weights{k} = L{k}' * delta{k} / n;
        d_biase{k} = mean(delta{k},1);
        % ���¶���
        v_weights{k} = momentum * v_weights{k} - lr * d_weights{k};
        v_biase{k} = momentum * v_biase{k} - lr * d_biase{k};
        % ���²���
        weights{k} = weights{k} + v_weights{k};
        biase{k} = biase{k} + v_biase{k};
    end

    % ��¼loss
    ce_loss(i+1) = loss(x_train,y_train,weights,biase);
    % ����Ӧѧϰ��,ѧϰ�ʵķ�Χ0.01~0.9
    if (ce_loss(i+1) <= opts.training_object)
        break
    end
    if(i>1)
        diff_loss = ce_loss(i) - ce_loss(i+1);
        % loss��������Сѧϰ�ʣ�����С��0.01
        if(diff_loss < -1e-3) % ����loss��΢С��������������ΪС��-1e3��������0
            lr = lr * 0.7;
            a = lr < 0.01;
            lr = 0.01*a + (1-a)*lr;    
        end
        % ǰ�ڵ�loss�仯̫С������ѧϰ�ʣ���������0.9
        if(0<diff_loss && diff_loss<1e-2)
            lr = lr * 1.05;
            b = lr > 0.9;
            lr = 0.9*b + (1-b)*lr;
        end
    end
    if(mod(i+1,300)==0); disp(['�����',num2str(i+1),'�ε���']);end
end

% ���ﵽ���������δ�ﵽѵ��Ŀ�꣬�򱨴�
if(ce_loss(i+1) > opts.training_object);warning('δ�ﵽѵ��Ŀ�ꣻ�������ø���Ĳ���,������Ż�����,���Сѵ��Ŀ�ꡣ');end

%% չʾѵ�����
initial_cross_entropy = loss(x_train,y_train,initial_para.weights,initial_para.biase);
final_loss = loss(x_train,y_train,weights,biase);
display(initial_cross_entropy);
display(final_loss);
    % loss����
plot(ce_loss(1:i+1))
    % ѵ������������ֵ��Ϊ0.5
% error_rate_of_TrainingSet0 = error_rate(x,y,initial_para.weights,initial_para.biase);
[probs_of_positve,error_rate_of_TestSet] = error_rate(x_test,y_test,weights,biase);
% display(error_rate_of_TrainingSet0);
display(error_rate_of_TestSet);

time_consume = toc;


end

%% ������������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% �� �� �� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% �������Ӻ���

function cross_entropy = loss(x,y,weights,biase)
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));
numoflayers = length(weights) + 1;
L = cell(1,numoflayers); % ���ڼ�¼MLP��������ֵ
L{1} = x;
for k = 2 : numoflayers
    L{k} = sigm(L{k-1},weights{k-1},biase{k-1});
end
cross_entropy = mean(-y .* log(L{numoflayers}) - (1-y) .* log(1-L{numoflayers}),1);
end

function square_error = loss2(x,y,weights,biase)
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));
numoflayers = length(weights) + 1;
L = cell(1,numoflayers); % ���ڼ�¼MLP��������ֵ
L{1} = x;
for k = 2 : numoflayers
    L{k} = sigm(L{k-1},weights{k-1},biase{k-1});
end
square_error = norm(L{numoflayers} - y);
end

function [y_pred,error_rate_of_TrainingSet] = error_rate(x,y,weights,biase)
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));
numoflayers = length(weights) + 1;
L = cell(1,numoflayers); % ���ڼ�¼MLP��������ֵ
L{1} = x;
for k = 2 : numoflayers
    L{k} = sigm(L{k-1},weights{k-1},biase{k-1});
end
y_pred = L{numoflayers}; % L3�������������㣬output layer
error_rate_of_TrainingSet = sum(abs((y_pred>0.5)-y)) / size(y,1);
end











