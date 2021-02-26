clc;
clear;

mainPath = "";

% Read Family Data
CaseBeacon = readtable(strcat(mainPath, "/CaseBeacon.txt"));
CaseBeacon = array2table(table2array(CaseBeacon(:, 2:end)).', 'VariableNames', CaseBeacon{:, 1}, 'RowNames', CaseBeacon.Properties.VariableNames(2:end));

ControlBeacon = readtable(strcat(mainPath, "/ControlBeacon.txt"));
ControlBeacon = array2table(table2array(ControlBeacon(:, 2:end)).', 'VariableNames', ControlBeacon{:, 1}, 'RowNames', ControlBeacon.Properties.VariableNames(2:end));

CasePeople = readtable(strcat(mainPath, "/CaseGenomes.txt"));
CasePeople = array2table(table2array(CasePeople(:, 2:end)).', 'VariableNames', CasePeople{:, 1}, 'RowNames', CasePeople.Properties.VariableNames(2:end));

ControlPeople = readtable(strcat(mainPath, "/ControlGenomes.txt"));
ControlPeople = array2table(table2array(ControlPeople(:, 2:end)).', 'VariableNames', ControlPeople{:, 1}, 'RowNames', ControlPeople.Properties.VariableNames(2:end));


% Load MAF, LD 
AFs = readtable(strcat(mainPath, "/MAF.txt"), 'Delimiter', ',' );
AFs = sortrows(AFs, 3);

LDScore1 = readtable(strcat(mainPath, "/ld_new20_CEU_07_sortedM.txt"));
LDs = digraph(LDScore1{:,4},LDScore1{:,5},LDScore1{:,11});
LDs = addedge(LDs, LDScore1{:,5},LDScore1{:,4},LDScore1{:,12});

NN = 60;
test_size = 20;
query_count = 100;

for control=[0:1]
    if control == 0 % Case People
        test_group = CasePeople;
    else % Control People
        test_group = ControlPeople;
    end
    Attack(CaseBeacon, test_group, query_count, AFs, LDs, false, 0, control);
    for t=["Optimal"]
        PlotPower(t, 0, query_count);
    end
end



