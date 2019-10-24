% LSTM for classification二分类任务（递归神经网络）一行就是一条序列，所有序列的长度可不同，用NaNt填补空值
function [Weights_of_LSTMunits,Weights_of_output,y_hat,error_rate_of_test,final_loss,time_consume] = modified_LSTMver2(seq_train,y_train,struc,seq_test,y_test,opts)
% Weights_of_LSTMunits是结构体，包含遗忘门、输入门、输出门的“所有”参数，也就是输入层到隐层之间的参数
% 例如，Weights_of_LSTMunits.forget_gate是(nx+nh+1)×nh的矩阵，依次下来对应的参数是[Ui; Wi; bi]
tic
rng(2018) % 愛衣ちゃん大勝利
nx1 = struc(1); nx2 = struc(2); nh = struc(3); 
x_train = seq_train(:,2:end); record_train = seq_train(:,1); 
warr_train = x_train(:,1:nx1); cust_train = x_train(:,end-nx2+1:end);
x_test = seq_test(:,2:end); record_test = seq_test(:,1);
warr_test = x_test(:,1:nx1); cust_test = x_test(:,end-nx2+1:end);
num_of_sample = size(x_train,1); ny = size(y_train,2); 
if(nargin == 5)
    opts.epoch = 3e3; 
    opts.learning_rate = 0.05; 
    opts.momentum = 0.9; 
    opts.training_object = 1e-2;     % 设置训练目标
    opts.batch_size = ceil(size(y_train,1)/40);
    opts.T = 9;
end
epoch = opts.epoch; lr = opts.learning_rate; momentum = opts.momentum; 
training_object = opts.training_object; batch_size = opts.batch_size;  T = opts.T;
% 参数初始化
coef = 0.1; coef_biase = 1;
W_i = [coef*randn(nx1+nh,nh); -coef_biase*rand(1,nh)]; W_a = [coef*randn(nx1+nh,nh); -coef_biase*rand(1,nh)]; % 输入门，输出门biase用负数初始化
W_o = [coef*randn(nx1+nh,nh); -coef_biase*rand(1,nh)]; W_f = [coef*randn(nx1+nh,nh); coef_biase*rand(1,nh)];  % 遗忘门biase用正数初始化
W_y = [coef * randn(nh+nx2,ny); zeros(1,ny)]; 
% 动量项初始化
vW_i = zeros(nx1+nh+1,nh); vW_a = zeros(nx1+nh+1,nh); vW_o = zeros(nx1+nh+1,nh); vW_f = zeros(nx1+nh+1,nh); 
vW_y = zeros(nh+nx2+1,ny); 

% 训练: 前向传播 -> 后向传播 -> 更新梯度/动量 -> 更新参数及学习率 -> 更新loss并调整学习率
remainder = mod(num_of_sample, batch_size);
start = 1 : batch_size : (num_of_sample-remainder);
final = batch_size : batch_size : (num_of_sample-remainder);
final(end) = final(end) + remainder;
batch_index = [start;final]'; 
loss = zeros(epoch,1); 
h0 = zeros(1,nh); c0 = zeros(1,nh); % 设置初始细胞状态为0
for i = 1 : epoch
    j = mod(i,size(batch_index,1)) + 1 ; %选batch
    dW_y = 0; dW_o = 0; dW_f = 0; dW_i = 0; dW_a = 0; 
    for n = batch_index(j,1):batch_index(j,2)
        seq_len = min([record_train(n)-2,T]) + 1;
        x_n = warr_train(n-seq_len+1:n,:);
        y_n = y_train(n,1);
        hidden = [h0; zeros(seq_len,nh)]; cell_state = [c0; zeros(seq_len,nh)];
        input_i = zeros(seq_len,nh); input_a = input_i; 
        forget = input_i; output = input_i; 
        % 前向传播
        for k = 1:seq_len
            x_n_k_ = [x_n(k,:) hidden(k,:) 1];
            input_i(k,:) = sigm(x_n_k_ * W_i); input_a(k,:) = 2*tanh(x_n_k_ * W_a);
            forget(k,:) = sigm(x_n_k_ * W_f); output(k,:) = sigm(x_n_k_ * W_o);
            cell_state(k+1,:) = forget(k,:).* cell_state(k,:) + input_i(k,:).*input_a(k,:);
            hidden(k+1,:) = output(k,:) .* tanh(cell_state(k+1,:));
        end
        h_n_end_ = [hidden(end,:),cust_train(n,:),1];
        y_hat = sigm(h_n_end_*W_y);
        
        % 后向传播(BPTT)：Part1更新V与W_o(usual BP); Part2更新W_i，W_f，W_a ( truncation RTRL)
        % Part 1: usual BP
        grad_W_y = h_n_end_' * (y_hat - y_n);
        delta_Eo = (y_hat - y_n) * W_y(1:nh,:)' .* tanh(cell_state(end,:)) .* (output(end,:).*(1-output(end,:)));
        grad_W_o = [x_n(end,:),hidden(end-1,:),1]' * delta_Eo;
        % Part 2: RTRL
        delta_Ec = (y_hat - y_n) * W_y(1:nh,:)' .* output(end,:) .* (1 - tanh(cell_state(end,:)).^2);
        delta_cW_f = zeros(nx1+nh+1,nh); delta_cW_i = zeros(nx1+nh+1,nh); delta_cW_a = zeros(nx1+nh+1,nh);
        for k = 1:seq_len   % 递归计算delta_cW
            x_n_k_ = [x_n(k,:) hidden(k,:) 1];
            delta_cW_f = x_n_k_' * (cell_state(k,:) .* forget(k,:) .* (1-forget(k,:))) + delta_cW_f * diag(forget(k,:));
            delta_cW_i = x_n_k_' * (input_a(k,:) .* input_i(k,:) .* (1 - input_i(k,:)))+ delta_cW_i * diag(forget(k,:));
            delta_cW_a = x_n_k_' * (input_i(k,:) .* (1 - tanh(input_a(k,:)).^2)) + delta_cW_a * diag(forget(k,:));
        end
        grad_W_f = delta_cW_f * diag(delta_Ec); grad_W_i = delta_cW_i * diag(delta_Ec); grad_W_a = delta_cW_a * diag(delta_Ec);
        
        dW_y = dW_y + grad_W_y; dW_o = dW_o + grad_W_o;
        dW_f = dW_f + grad_W_f; dW_i = dW_i + grad_W_i; dW_a = dW_a + grad_W_a;
    end
    % 更新参数
    dW_y = dW_y / batch_size; dW_o = dW_o / batch_size; 
    dW_f = dW_f / batch_size; dW_i = dW_i / batch_size; dW_a = dW_a / batch_size; 
    
    vW_y = momentum * vW_y - lr * dW_y; vW_o = momentum * vW_o - lr * dW_o; 
    vW_f = momentum * vW_f - lr * dW_f; vW_i = momentum * vW_i - lr * dW_i; vW_a = momentum * vW_a - lr * dW_a; 
    
    W_y = W_y + vW_y; W_o = W_o + vW_o; 
    W_f = W_f + vW_f; W_i = W_i + vW_i; W_a = W_a + vW_a; 

    % 计算loss
%     loss(i) = predict(warr_train,cust_train,record_train,y_train,nh,W_f,W_i,W_a,W_o,W_y,T);
%     if(loss(i) <= training_object); break; end % 达到训练目标，停止迭代
%     if(i > 1)
%         diff_loss = loss(i-1) - loss(i);
%         if(diff_loss < -1e-3); lr = lr * 0.7; a = lr<0.01; lr = 0.01*a + (1-a)*lr; end
%         if(0<diff_loss && diff_loss<1e-2); lr = lr * 1.05; b = lr>0.9; lr = 0.9*b + (1-b)*lr; end
%     end
    if(mod(i,300)==0); disp(['已完成',num2str(i),'次迭代']); end
end

% plot(loss(1:i))


[final_loss,y_hat] =  predict(warr_test,cust_test,record_test,y_test,nh,W_f,W_i,W_a,W_o,W_y,T);
act_y_test = y_test(record_test~=1);
error_rate_of_test = sum(abs((y_hat>0.5)-act_y_test)) / size(act_y_test,1);
disp(final_loss)

Weights_of_LSTMunits.input_gate_i = W_i; Weights_of_LSTMunits.input_gate_a = W_a; 
Weights_of_LSTMunits.forget_gate = W_f; Weights_of_LSTMunits.output_gate_i = W_o; 
Weights_of_output = W_y;
time_consume = toc;
end

function [ce_loss,y_hat] = predict(warr_train,cust_train,record_train,y_train,nh,W_f,W_i,W_a,W_o,W_y,T)
y_hat = zeros(size(y_train,1)-sum(record_train==1),1); 
act_y_train = y_train(record_train~=1);
i = 1;
for n = 1:size(y_train,1)
    seq_len = min([record_train(n)-2,T]) + 1;
    x_n = warr_train(n-seq_len+1:n,:); 
    hidden = zeros(1,nh); cell_state = zeros(1,nh);  
    for k = 1:seq_len
        x_n_k_ = [x_n(k,:) hidden 1];
        input_i = sigm(x_n_k_ * W_i); input_a = 2*tanh(x_n_k_ * W_a);
        forget = sigm(x_n_k_ * W_f); output = sigm(x_n_k_ * W_o);
        cell_state = forget.* cell_state + input_i.*input_a;
        hidden = output .* tanh(cell_state);
    end
    h_n_end_ = [hidden,cust_train(n,:),1];
    y_hat(i) = sigm(h_n_end_*W_y); i = i + 1;
end
ce_loss = mean(-act_y_train.*log(y_hat) - (1-act_y_train).*log(1-y_hat),1);
end


function sigmoid = sigm(net)
sigmoid = 1 ./ (1 + exp(-net)); 
end
