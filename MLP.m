% BP算法训练多层(任意层)感知器
% 该MLP的模型参数：连接权重weights, 偏置项biase
% 该MLP用于二分类任务，故输出层采用logistic。此外，其余各层的激活函数也为logistic
function [weights,biase,probs_of_positve,error_rate_of_TestSet,final_loss,time_consume] = MLP(x_train,y_train,MLP_struc,x_test,y_test,initial_para,opts)
%% 优化参数、模型初始参数的缺省设置
tic
rng(2018)
if(nargin < 3);error('输入参数不能少于3个！');end
num_of_input = MLP_struc(1); num_of_output = MLP_struc(end);
if(size(x_train,2) ~= num_of_input || size(y_train,2) ~= num_of_output)
    error('神经网络的输入(出)层神经元个数与特征数(标签数)不同！')
end
len = length(MLP_struc);
if(len>5);warning('层数较多，训练时间长!建议计算运行时间，并做好耐心等待的准备');end
if(nargin == 5)   
    % 定义优化参数的缺省设置
    opts.epoch = 3e3;
    opts.learning_rate = 0.05;
    opts.batch_size = ceil(size(x_train,1)/40);
    opts.momentum = 0.9;   
    % 设置训练目标：loss小于这个阈值时，就停止迭代
    opts.training_object = 1e-4;
    % 初始参数缺省设置
    coef = 0.1;
    for i = 1 : len-1
       initial_para.weights{i} = coef*randn(MLP_struc(i),MLP_struc(i+1));
       initial_para.biase{i} = zeros(1,MLP_struc(i+1));
    end
  
end

%% 设置MLP的初始训练参数
weights = initial_para.weights; biase = initial_para.biase; 

    %初始动量，设置为0; 初始梯度设置为0。
v_weights = cell(1,len-1); v_biase = cell(1,len-1); 
d_weights = cell(1,len-1); d_biase = cell(1,len-1);
for i = 1 : len-1
    v_weights{i} = zeros(size(initial_para.weights{i}));
    v_biase{i} = zeros(size(initial_para.biase{i}));
    d_weights{i} = zeros(size(initial_para.weights{i}));
    d_biase{i} = zeros(size(initial_para.biase{i}));
end

L = cell(1,len); % 用于记录MLP各层的输出值
delta = cell(1,len-1); % 用于记录反向传播的输出值
ce_loss = zeros(opts.epoch,1); % 用于记录交叉熵的变化
lr = opts.learning_rate; momentum = opts.momentum;

%% 定义激活函数，logistic函数
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));

%% 使用小批量的动量梯度下降法训练 
    % 先分好batch
num_of_sample = size(x_train,1);
remainder = mod(num_of_sample, opts.batch_size);
start = 1 : opts.batch_size : (num_of_sample-remainder);
final = opts.batch_size : opts.batch_size : (num_of_sample-remainder);
final(end) = final(end) + remainder;
batch_index = [start;final]';
    % 开始训练
for i = 0 : opts.epoch-1
    j = mod(i,size(batch_index,1)) + 1 ; %选batch
    
    % forward propagation前向传播，计算各层输出，L1~Ln，共n层
    L{1} = x_train(batch_index(j,1):batch_index(j,2),:);
    for k = 2 : len
        L{k} = sigm(L{k-1},weights{k-1},biase{k-1});
    end
    
    % back propagation 反向传播，计算"梯度"delta3, delta2, delta1
    target = y_train(batch_index(j,1):batch_index(j,2),:); % 真实标记
    %dE_wrt_Ln = -target./L{len} + (1-target)./(1-L{len}); % 交叉熵损失函数对输出层的偏导数,这个式子是不稳定的，可能会出现NaN; 
    % dE_wrt_Ln = L{len} - target; % 这是二次损失函数对输入层的偏导数（用于回归任务）
        % 误差反向传播
    % delta{len-1} = dE_wrt_Ln .* L{len} .* (1-L{len});
    delta{len-1} = L{len} - target; % 这样写就不会出现NaN了
    for k = (len-2):-1:1  
        delta{k} = delta{k+1} * weights{k+1}' .* L{k+1} .* (1-L{k+1});
    end
    
    % 计算梯度,并更新参数
    n = size(L{1},1);
    for k = (len-1):-1:1  % 为了“BP”的方向保持一致，写成len:1,但写成1:len也是对的
        % 更新梯度
        d_weights{k} = L{k}' * delta{k} / n;
        d_biase{k} = mean(delta{k},1);
        % 更新动量
        v_weights{k} = momentum * v_weights{k} - lr * d_weights{k};
        v_biase{k} = momentum * v_biase{k} - lr * d_biase{k};
        % 更新参数
        weights{k} = weights{k} + v_weights{k};
        biase{k} = biase{k} + v_biase{k};
    end

    % 记录loss
    ce_loss(i+1) = loss(x_train,y_train,weights,biase);
    % 自适应学习率,学习率的范围0.01~0.9
    if (ce_loss(i+1) <= opts.training_object)
        break
    end
    if(i>1)
        diff_loss = ce_loss(i) - ce_loss(i+1);
        % loss上升，缩小学习率，但不小于0.01
        if(diff_loss < -1e-3) % 允许loss有微小的增大，这里设置为小于-1e3，而不是0
            lr = lr * 0.7;
            a = lr < 0.01;
            lr = 0.01*a + (1-a)*lr;    
        end
        % 前期的loss变化太小，增大学习率，但不大于0.9
        if(0<diff_loss && diff_loss<1e-2)
            lr = lr * 1.05;
            b = lr > 0.9;
            lr = 0.9*b + (1-b)*lr;
        end
    end
    if(mod(i+1,300)==0); disp(['已完成',num2str(i+1),'次迭代']);end
end

% 若达到最大步数后，仍未达到训练目标，则报错
if(ce_loss(i+1) > opts.training_object);warning('未达到训练目标；建议设置更大的步数,或更改优化参数,或调小训练目标。');end

%% 展示训练结果
initial_cross_entropy = loss(x_train,y_train,initial_para.weights,initial_para.biase);
final_loss = loss(x_train,y_train,weights,biase);
display(initial_cross_entropy);
display(final_loss);
    % loss曲线
plot(ce_loss(1:i+1))
    % 训练集误差，分类阈值设为0.5
% error_rate_of_TrainingSet0 = error_rate(x,y,initial_para.weights,initial_para.biase);
[probs_of_positve,error_rate_of_TestSet] = error_rate(x_test,y_test,weights,biase);
% display(error_rate_of_TrainingSet0);
display(error_rate_of_TestSet);

time_consume = toc;


end

%% 上面是主函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% 分 界 线 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 下面是子函数

function cross_entropy = loss(x,y,weights,biase)
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));
numoflayers = length(weights) + 1;
L = cell(1,numoflayers); % 用于记录MLP各层的输出值
L{1} = x;
for k = 2 : numoflayers
    L{k} = sigm(L{k-1},weights{k-1},biase{k-1});
end
cross_entropy = mean(-y .* log(L{numoflayers}) - (1-y) .* log(1-L{numoflayers}),1);
end

function square_error = loss2(x,y,weights,biase)
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));
numoflayers = length(weights) + 1;
L = cell(1,numoflayers); % 用于记录MLP各层的输出值
L{1} = x;
for k = 2 : numoflayers
    L{k} = sigm(L{k-1},weights{k-1},biase{k-1});
end
square_error = norm(L{numoflayers} - y);
end

function [y_pred,error_rate_of_TrainingSet] = error_rate(x,y,weights,biase)
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));
numoflayers = length(weights) + 1;
L = cell(1,numoflayers); % 用于记录MLP各层的输出值
L{1} = x;
for k = 2 : numoflayers
    L{k} = sigm(L{k-1},weights{k-1},biase{k-1});
end
y_pred = L{numoflayers}; % L3即是网络的输出层，output layer
error_rate_of_TrainingSet = sum(abs((y_pred>0.5)-y)) / size(y,1);
end











