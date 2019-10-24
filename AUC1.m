function [auc_result,FPR,TPR] = AUC1(target,probs_of_positive) 
[~ ,I] = sort(probs_of_positive,'descend');
label = target(I);
TPR = zeros(size(target)); FPR = zeros(size(target)); 
for i = 1 :length(target)
    TPR(i) = sum(label(1:i) == 1) / sum(label == 1);
    FPR(i) = sum(label(1:i) == 0) / sum(label == 0);
end
plot(FPR,TPR)
axis square
set(gca,'XTick',0:0.2:1); set(gca,'YTick',0:0.2:1);
xlabel('FPR'); ylabel('TPR'); title('ROC curve')
auc_result = trapz(FPR,TPR);
disp(auc_result);
text(0.5,0.5,['AUC = ',sim2str(vpa(auc_result,3))])

end



