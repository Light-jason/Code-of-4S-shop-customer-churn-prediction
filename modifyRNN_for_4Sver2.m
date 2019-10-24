% RNN for 二分类任务（递归神经网络）
% modifyRNN_for_4Sver2的数据集去掉了record=1
function [weights,y_hat,error_rate,final_loss,time_consume] = modifyRNN_for_4Sver2(seq_train,y_train,struc,seq_test,y_test,opts)
tic
rng(2018)
num_of_input1 = struc(1); num_of_input2 = struc(2); num_of_hid = struc(3); 
x_train = seq_train(:,2:end); record_train = seq_train(:,1); 
warr_train = x_train(:,1:num_of_input1); cust_train = x_train(:,num_of_input1+1:end);
x_test = seq_test(:,2:end); record_test = seq_test(:,1);
warr_test = x_test(:,1:num_of_input1); cust_test = x_test(:,num_of_input1+1:end);
num_of_sample = size(x_train,1); num_of_output = size(y_train,2); 
if(nargin == 5)
    opts.epoch = 3e3; 
    opts.learning_rate = 0.05; 
    opts.momentum = 0.9; 
    opts.training_object = 1e-2;     % 设置训练目标
    opts.batch_size = ceil(size(y_train,1)/40); 
    opts.T = 9;   % 这里的T是向上取特征的最大个数,相当于RNN_for_4S中T=4的情形
end
if(nargin == 3); num_of_hid = ceil((num_of_input1 + num_of_output)/2); end
epoch = opts.epoch; lr = opts.learning_rate; momentum = opts.momentum; 
training_object = opts.training_object; batch_size = opts.batch_size; T = opts.T;
% 权重初始化
coeff = 0.1;
U = coeff * randn(num_of_input1,num_of_hid); W = coeff * randn(num_of_hid); b_h = zeros(1,num_of_hid); 
V = coeff * randn(num_of_hid+num_of_input2,num_of_output); b_y = zeros(1,num_of_output); 
% 动量项初始化为0
vU = zeros(size(U)); vW = zeros(size(W)); vb_h = zeros(size(b_h)); 
vV = zeros(size(V)); vb_y = zeros(size(b_y));   

% 训练: 前向传播 -> 后向传播 -> 更新梯度/动量 -> 更新参数及学习率 -> 更新loss并调整学习率
remainder = mod(num_of_sample, batch_size);
start = 1 : batch_size : (num_of_sample-remainder);
final = batch_size : batch_size : (num_of_sample-remainder);
final(end) = final(end) + remainder;
batch_index = [start;final]'; 
loss = zeros(epoch,1); h0 = sigm(0*b_h);
for i = 1 : epoch
    j = mod(i,size(batch_index,1)) + 1 ; %选batch
    dV = 0; db_h = 0; dW = 0; dU = 0; db_y = 0; 
    for n = batch_index(j,1):batch_index(j,2)
        seq_len = min([record_train(n)-2,T]) + 1;
        x_n = warr_train(n-seq_len+1:n,:); 
        y_n = y_train(n,1);
        h_n = zeros(seq_len,num_of_hid); 
        % 前向传播
        for k = 1:seq_len
            x_n_k = x_n(k,:);
            if(k==1); h_n_k = sigm(x_n_k*U + h0*W + b_h); h_h_k1 = h_n_k; else; h_n_k = sigm(x_n_k*U + h_h_k1*W + b_h); h_h_k1 = h_n_k;end
            h_n(k,:) = h_n_k;
        end
        h_n_end = h_n(end,:); hidden = [h_n_end,cust_train(n,:)];
        y_hat = sigm(hidden*V + b_y);
        
        % 后向传播(BPTT)
        grad_V = hidden' * (y_hat - y_n); grad_b_y = y_hat - y_n;
        grad_W = 0; grad_U = 0; grad_b_h = 0;
        delta_k = (y_hat - y_n)*V(1:num_of_hid,:)' .* h_n_end .* (1-h_n_end);
        for k = seq_len:-1:1
            x_n_k = x_n(k,:);
            if(k==1); h_n_k1 = h0; else; h_n_k1 = h_n(k-1,:);end
            grad_W = grad_W + h_n_k1'*delta_k; grad_U = grad_U + x_n_k'*delta_k; grad_b_h = grad_b_h + delta_k;
            delta_k = delta_k * W' .* h_n_k1 .* (1-h_n_k1);
        end
        dV = dV + grad_V; db_y = db_y + grad_b_y;
        dW = dW + grad_W; dU = dU + grad_U; db_h = db_h + grad_b_h;
        % 这里可设置threshold，以防梯度爆炸
    end
    % 更新参数
    dV = dV / batch_size; db_y = db_y / batch_size;
    dW = dW / batch_size; dU = dU / batch_size; db_h = db_h / batch_size;
    
    vV = momentum * vV - lr * dV; vb_y = momentum * vb_y - lr * db_y;
    vW = momentum * vW - lr * dW; vU = momentum * vU - lr * dU; vb_h = momentum * vb_h - lr * db_h;
    V = V + vV; b_y = b_y + vb_y; W = W + vW; U = U + vU; b_h = b_h + vb_h;
    % 计算loss
%     loss(i) = cross_entropy(warr_train,cust_train,record_train,y_train,U,W,b_h,V,b_y,T);
%     if(loss(i) <= training_object); break; end % 达到训练目标，停止迭代
%     if(i > 1)
%         diff_loss = loss(i-1) - loss(i);
%         if(diff_loss < -1e-3); lr = lr * 0.7; a = lr<0.01; lr = 0.01*a + (1-a)*lr; end
%         if(0<diff_loss && diff_loss<1e-2); lr = lr * 1.05; b = lr>0.9; lr = 0.9*b + (1-b)*lr; end
%     end
    if(mod(i,300)==0); disp(['已完成',num2str(i),'次迭代']);end
end

% plot(loss(1:i))
if(loss(i) > training_object);warning('未达到训练目标；建议设置更大的步数,或更改优化参数,或调小训练目标。');end 
[final_loss,y_hat] = cross_entropy(warr_test,cust_test,record_test,y_test,U,W,b_h,V,b_y,T);
act_y_test = y_test(record_test~=1);
error_rate = sum(abs((y_hat>0.5)-act_y_test)) / size(act_y_test,1);
weights.U = U; weights.W = W; weights.b_h = b_h; 
weights.V = V; weights.b_y = b_y;
time_consume = toc;
end


function [ce_loss,y_hat] = cross_entropy(warr_train,cust_train,record_train,y_train,U,W,b_h,V,b_y,T)
h0 = sigm(0*b_h); 
y_hat = zeros(size(y_train,1)-sum(record_train==1),1); 
act_y_train = y_train(record_train~=1);
i = 1;
% 前向传播
for n = 1:size(y_train,1)
    seq_len = min([record_train(n)-2,T]) + 1;
    x_n = warr_train(n-seq_len+1:n,:); 
    % 前向传播
    for k = 1:seq_len
        x_n_k = x_n(k,:);
        if(k==1); h_n_k = sigm(x_n_k*U + h0*W + b_h); h_h_k1 = h_n_k; else; h_n_k = sigm(x_n_k*U + h_h_k1*W + b_h); h_h_k1 = h_n_k;end
    end
    hidden = [h_n_k,cust_train(n,:)];
    y_hat(i) = sigm(hidden*V + b_y); i = i + 1;
end
ce_loss = mean(-act_y_train.*log(y_hat) - (1-act_y_train).*log(1-y_hat),1);
end


function ht = sigm(net)
ht = 1 ./ (1 + exp(-net)); 
end