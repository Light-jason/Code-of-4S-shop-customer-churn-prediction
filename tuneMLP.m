tic
% data preparation for logistic regression & SVM
data = csvread('dataset1.csv',1);
recordID = data(:,1:2); driverID = recordID(recordID(:,1)~=1,2);
    % 构造一步预测数据集
t = 1; % “自回归”使用的特征个数
records = data(:,1); numofrecords = length(records); 
numofID = length(unique(data(:,2))); 
warranty1 = data(:,4:5);warranty2 = data(:,6:50); costomer_demography = data(:,51:end); 
s1 = size(warranty1,2); s2 = size(warranty2,2); s = s1+s2; c = size(costomer_demography,2);
dataset_for_LR = zeros(numofrecords-numofID,t*s+c+2);
k = 1;
for i = 1:numofrecords
    if(records(i) ~= 1)
        label = data(i,2:3); 
        feature_i = zeros(1,t*s);
        for j = 1:min([t,records(i)-1])
            feature_i(1,s*(j-1)+1:s*j) = [warranty1(i-j+1,:),warranty2(i-j,:)]; 
        end
        feature__i = [feature_i,costomer_demography(i,:)];
        sample_i = [label,feature__i];
        dataset_for_LR(k,:) = sample_i; k = k + 1;
    end
end
y = dataset_for_LR(:,2); x = dataset_for_LR(:,3:end); 

% 10-fold CV分组
batch_size = 1536; num_of_sample=15363;
remainder = mod(num_of_sample, batch_size);
start = 1 : batch_size : (num_of_sample-remainder);
final = batch_size : batch_size : (num_of_sample-remainder);
final(end) = final(end) + remainder;
batch_index = [start;final]';
y_pred_prob = []; 
cross_entropy = zeros(10,1); TimeConsume = zeros(10,1);
for k = 1:10
    test_startID = batch_index(k,1); test_endID = batch_index(k,2); 
    x_test = x((driverID>=test_startID)&(driverID<=test_endID),:); y_test = y((driverID>=test_startID)&(driverID<=test_endID),:); 
    x_train = x(~((driverID>=test_startID)&(driverID<=test_endID)),:); y_train = y(~((driverID>=test_startID)&(driverID<=test_endID)),:);
    [~,~,probs_of_positive,~,final_loss,time_consume] = MLP(x_train,y_train,[158,30,1],x_test,y_test);
    y_pred_prob = [y_pred_prob; probs_of_positive]; 
    cross_entropy(k) = final_loss; TimeConsume(k) = time_consume; 
    disp(final_loss)
    disp(['已完成第',num2str(k),'次tune'])
end

% 测试结果
auc_result = AUC1(y,y_pred_prob);
threshold = 0.81;
error_rate_of_TestSet = sum(abs((y_pred_prob>threshold)-y)) / size(y,1);
Pred_Actu = [y_pred_prob>threshold,y];
PA = sum(Pred_Actu .* repmat([1 2],size(y,1),1),2);
TN = sum(PA==0); FN = sum(PA==1); FP = sum(PA==2); TP = sum(PA==3); 
confuse_matrix = [TP,FP;FN,TN];
disp(['10-foldCV的错误率:',num2str(error_rate_of_TestSet)])

RESULT_of_MLP = cell(4,1); resultMLP = [y,y_pred_prob];
RESULT_of_MLP{1} = resultMLP; RESULT_of_MLP{2} = auc_result; 
RESULT_of_MLP{3} = confuse_matrix; RESULT_of_MLP{4} = error_rate_of_TestSet;
RESULT_of_MLP{5} = cross_entropy; RESULT_of_MLP{6} = TimeConsume; 

save('RESULT_of_MLP.mat','RESULT_of_MLP')

toc

