clc;
clear;
close all;

ticket = zeros(3,9);
valid = false;

while valid == false

    position = zeros(3,9);

    for i = 1:3
        p = randperm(9,5);
        position(i,p) = 1;
    end

    count = sum(position);

    if all(count >= 1) && all(count <= 3)
        valid = true;
    end

end

for j = 1:9

    if j == 1
        range = 1:9;
    elseif j == 2
        range = 10:19;
    elseif j == 3
        range = 20:29;
    elseif j == 4
        range = 30:39;
    elseif j == 5
        range = 40:49;
    elseif j == 6
        range = 50:59;
    elseif j == 7
        range = 60:69;
    elseif j == 8
        range = 70:79;
    else
        range = 80:90;
    end

    rows = find(position(:,j) == 1);

    selected = range(randperm(length(range),length(rows)));
    selected = sort(selected);

    for k = 1:length(rows)
        ticket(rows(k),j) = selected(k);
    end

end

fprintf('\n');
fprintf('=============================\n');
fprintf('       TAMBOLA TICKET        \n');
fprintf('=============================\n');

for i = 1:3
    for j = 1:9

        if ticket(i,j) == 0
            fprintf('   -   ');
        else
            fprintf('%4d  ',ticket(i,j));
        end

    end
    fprintf('\n');
end

fprintf('=============================\n');