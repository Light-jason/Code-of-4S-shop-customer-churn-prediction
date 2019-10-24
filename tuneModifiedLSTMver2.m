% data prepareation for modify RNN ver2
tic
data = csvread('E:\����ʦ������\���ݺʹ���\1 ���ݼ�\dataset1.csv',1);
% ����һ��Ԥ�����ݼ�
t = 1; % ���Իع顱ʹ�õ���������,t=1ʱ��dataprepare_forRNN��һ����
records = data(:,1); numofrecords = length(records); 
numofID = length(unique(data(:,2))); 
warranty1 = data(:,4:5);warranty2 = data(:,6:50); costomer_demography = data(:,51:end); 
s1 = size(warranty1,2); s2 = size(warranty2,2); s = s1+s2; c = size(costomer_demography,2);
dataset_for_RNN = zeros(numofrecords-numofID,t*s+c+3); 
k = 1;
for i = 1:numofrecords
    if(records(i) ~= 1)
        label = data(i,2:3); 
        feature_i = zeros(1,t*s);
        for j = 1:min([t,records(i)-1])
            feature_i(1,s*(j-1)+1:s*j) = [warranty1(i-j+1,:),warranty2(i-j,:)]; 
        end
        feature__i = [feature_i,costomer_demography(i,:)];
        sample_i = [records(i,1),label,feature__i];
        dataset_for_RNN(k,:) = sample_i; k = k + 1;
    end
end
driverID = dataset_for_RNN(:,2); dataset_for_RNN(:,2) = []; y = dataset_for_RNN(:,2);
% 10-fold CV����
batch_size = 1536; num_of_sample=15363;
remainder = mod(num_of_sample, batch_size);
start = 1 : batch_size : (num_of_sample-remainder);
final = batch_size : batch_size : (num_of_sample-remainder);
final(end) = final(end) + remainder;
batch_index = [start;final]';
y_pred_prob = []; 
cross_entropy = zeros(10,1); TimeConsume = zeros(10,1);
for k = 10
    test_startID = batch_index(k,1); test_endID = batch_index(k,2);
    data_test = dataset_for_RNN((driverID>=test_startID)&(driverID<=test_endID),:);
    data_train = dataset_for_RNN(~((driverID>=test_startID)&(driverID<=test_endID)),:);
    y_train = data_train(:,2); data_train(:,2) = []; seq_train = data_train;
    y_test = data_test(:,2); data_test(:,2) = []; seq_test = data_test;
    [~,~,y_hat,~,final_loss,time_consume] = modified_LSTMver2(seq_train,y_train,[111,0,30],seq_test,y_test);  % LSTM
	%[~,~,y_hat,~,final_loss,time_consume] = modified_LSTMver2(seq_train,y_train,[47,64,30],seq_test,y_test);% LSTM-2L
    y_pred_prob = [y_pred_prob;y_hat];
    cross_entropy(k) = final_loss; TimeConsume(k) = time_consume;
    disp(final_loss)
    auc_result_per = AUC1(y_test,y_hat);
    disp(['����ɵ�',num2str(k),'��tune'])
end

auc_result = AUC1(y,y_pred_prob);
threshold = 0.81;
error_rate_of_TestSet = sum(abs((y_pred_prob>threshold)-y)) / size(y,1);
Pred_Actu = [y_pred_prob>threshold,y];
PA = sum(Pred_Actu .* repmat([1 2],size(y,1),1),2);
TN = sum(PA==0); FN = sum(PA==1); FP = sum(PA==2); TP = sum(PA==3); 
confuse_matrix = [TP,FP;FN,TN];
disp(['10-foldCV�Ĵ�����:',num2str(error_rate_of_TestSet)])

RESULT_of_2StepModifiedLSTM = cell(6,1); resultLR = [y,y_pred_prob];
RESULT_of_2StepModifiedLSTM{1} = resultLR; RESULT_of_2StepModifiedLSTM{2} = auc_result; 
RESULT_of_2StepModifiedLSTM{3} = confuse_matrix; RESULT_of_2StepModifiedLSTM{4} = error_rate_of_TestSet;
RESULT_of_2StepModifiedLSTM{5} = cross_entropy; RESULT_of_2StepModifiedLSTM{6} = TimeConsume; 


toc