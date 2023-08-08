clc
clear 


x = [1 2 3 4 5 6 7 ];
Training = readtable("training.csv");
A = Training(:,x);

Class_A = table2array(Training(:,end));
T = table2array(A);

Training.Properties.VariableNames=["Pregnancies","Glucose","Blood Pressure","Skin Thickness","BMI","DPF","Age","Outcome"];
writetable(Training,'TrainingOriginal.csv');

% % quante volte ogni osservazione verrà usata come base di sintesi
 obs_0 = 4*(Class_A==0);
 obs_1 = 8*(Class_A==1);
 obs = obs_0 + obs_1; 

% richiamo l'algoritmo SMOTE come implementato nella funzione apposita che
 [B,C,Xn,Cn] = smote(T,[],3, 'Class',Class_A,'SynthObs',obs); 
%[B,C,Xn,Cn] = smote(T,[],5, 'Class',Class_A); 

% ------------------------------------------------------- %
% verifico se sono riuscita a ribilanciare il training set



% calcolo le percentuali delle due classi
perc_uniSMOTE = (sum(C==1) *100)/(height(C));     
perc_zeriSMOTE = (sum(C==0) *100)/(height(C));   

disp('percentuale classe = 1')
disp(perc_uniSMOTE )
disp('percentuale classe = 0')
disp(perc_zeriSMOTE )


% arrotondamento vettori
% PREG
vettore1Preg = B(:,1);
vettore2Preg = unique(T(:,1));

% Arrotondamento dei dati di vettore1 al valore più simile di vettore2
vettArrPreg = zeros(size(vettore1Preg));

for i = 1:length(vettore1Preg)
    dato = vettore1Preg(i);
    [~, indice_simile] = min(abs(vettore2Preg - dato));
    vettArrPreg(i) = vettore2Preg(indice_simile);
end

%AGE 
vettore1Age = B(:,7);
vettore2Age = unique(T(:,7));

clear dato indice_simile i
vettArrAge = zeros(size(vettore1Age));

for i = 1:length(vettore1Age)
    dato = vettore1Age(i);
    [~, indice_simile] = min(abs(vettore2Age - dato));
    vettArrAge(i) = vettore2Age(indice_simile);
end

B(:,1) = vettArrPreg;
B(:,7) = vettArrAge;


% ----------------------------------------------------- %
% scrittura del file con il nuovo dataset

F = [B,C];
F = array2table(F);




Ori = table2array(F(1:532,:));
Aug=table2array(F(533:end,:));

OriTest0=Ori(Ori(:,8)==0,:);
OriTest1=Ori(Ori(:,8)==1,:);

AugTest0=Aug(Aug(:,8)==0,:);
AugTest1=Aug(Aug(:,8)==1,:);


for i = 1:7

    H_DataAug0(i) = kstest2(OriTest0(:,i),AugTest0(:,i));
    H_DataAug1(i) = kstest2(OriTest1(:,i),AugTest1(:,i));

end

F.Properties.VariableNames=["Feature 1","Feature 2","Feature 3","Feature 4","Feature 6","Feature 7","Feature 8","Feature 9"];
writetable(F,'AugmentedTrainingRng6.csv');


disp('# classe = 1')
disp(sum(C==1))
disp('# classe = 0')
disp(sum(C==0))


% %% ---------------------------------------------------------------------- %
% % Data Augmentation
% 
% % quante volte ogni osservazione verrà usata come base di sintesi
% obs_0 = 10*(strcmp(Class_A,'0'));
% obs_1 = 20*(strcmp(Class_A,'1'));
% obs = obs_0 + obs_1; 
% 
% % richiamo l'algoritmo SMOTE come implementato nella funzione apposita che
% [B1,C1,Xn1,Cn1] = smote(T,[],3, 'Class',Class_A,'SynthObs',obs); 
% % [B1,C1,Xn1,Cn1] = smote(T,[],3, 'Class',Class_A); 
% 
% % ------------------------------------------------------- %
% % verifico se sono riuscita a ribilanciare il training set
% 
% % ZeriSMOTE1 = sum(strcmp(C1,'0'));
% % UniSMOTE1 = sum(strcmp(C1,'1'));
% % 
% % % calcolo le percentuali delle due classi
% % perc_uniSMOTE1 = (UniSMOTE1 *100)/(height(C1));     
% % perc_zeriSMOTE1 = (ZeriSMOTE1 *100)/(height(C1));   
% % 
% % disp('percentuale classe = 1')
% % disp(perc_uniSMOTE1 )
% % disp('percentuale classe = 0')
% % disp(perc_zeriSMOTE1 )
% % 
% % % ora sono bilanciate 50 e 50
% % % ma anche aumentate
% 
% 
% % -------------------------------------------------------------------- %
% % i dati di Pregnancies e Age devono essere sistemati
% % vettore 1 da arrotondare
% % vettore 2 riferimento per arrotodarli
% 
% clear vettArrAge vettArrPreg vettore1Age vettore1Preg vettore2Age vettore2Preg
% clear dato i indice_simile
% 
% vettore1Preg = B1(:,1);
% vettore2Preg = unique(T(:,1));
% vettore1Age = B1(:,8);
% vettore2Age = unique(T(:,8));
% 
% % Arrotondamento dei dati di vettore1 al valore più simile di vettore2
% vettArrPreg = zeros(size(vettore1Preg));
% vettArrAge = zeros(size(vettore1Age));
% 
% for i = 1:length(vettore1Preg)
%     dato = vettore1Preg(i);
%     [~, indice_simile] = min(abs(vettore2Preg - dato));
%     vettArrPreg(i) = vettore2Preg(indice_simile);
% 
%     dato1 = vettore1Age(i);
%     [~, indice_simile1] = min(abs(vettore2Age - dato1));
%     vettArrAge(i) = vettore2Age(indice_simile1);
% end
% 
% 
% B1(:,1) = vettArrPreg;
% B1(:,8) = vettArrAge;
% 
% 
% %% ----------------------------------------------------- %
% % scrittura del file con il nuovo dataset
% 
% F1 = [num2cell(B1),C1];
% F1 = array2table(F1);
% 
% F1.Properties.VariableNames=["Pregnancies","Glucose","Blood Pressure","Skin Thickness","Insuline","BMI","DPF","Age","Outcome"];
% writetable(F1,'TrainingAugm.csv');
%%
A=[4.6	 1.5;...
4.3 1.3 ;...
5.8 2.2 ;...
5.9 1.8];
C=[0 0 1 1];
Q=squareform(pdist(A,"euclidean"));