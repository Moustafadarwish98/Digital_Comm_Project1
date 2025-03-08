close all
realization_num = 500;          % Number of realizations
samples_num = 7;                % Number of samples
ts = 0.01;                      % Sampling time of the DAC = 10 ms
tb = 0.07;                      % Bit width
total_t = tb/ts;                % Total bit time = tb/ts = 7 seconds
check_zero = 0;                 % Counter for zero elements in RZ
check_amplitude = 0;            % Counter for amplitude elements in RZ
t = 0.01 : ts : total_t;        % Time vector from 0.01s to 7s with 0.01s step size 
A = 4;                          % Amplitude
%% Random stream of bits for 500 realizations
data = randi([0 1], 500, 101);  % Generate a 500x101 matrix of random bits
                                % each row represents one realization of 101 bits
%% Random delay for realizations
delay = randi([0 6], 500, 1);    % Generate random delays between 0 and 6 samples
                                 % Each realization starts at a random delay
                                 % between 0 and 6 samples.
%% Polar NRZ: Mapping for 0 to be –A, 1 to be A
POLAR_NRZ = ((2*data)-1)*A;      % 0 maps to -A, 1 maps to A
UNIPOLAR_NRZ = data * A;         % 0 maps to 0, 1 maps to A
POLAR_RZ = ((2*data)-1)*A;       % 0 maps to 0, 1 maps to A
%% Samples generation
POLAR_NRZ_samples = repelem(POLAR_NRZ, 1, samples_num);  %converts each bit into a sampled waveform.
UNI_NRZ_samples = repelem(UNIPOLAR_NRZ, 1, samples); 
POLAR_RZ_samples = repelem(POLAR_RZ, 1, samples); 
%% Return to zero logic
POLAR_return_to_zero = POLAR_RZ_samples;
for counter_realization = 1 : realization_num
    for counter_samples = 7 : 707
        if check_zero ~= 3 
            POLAR_return_to_zero(counter_realization,counter_samples) = 0;
            check_zero = check_zero + 1;
        else
            check_amplitude = check_amplitude + 1;
        end
        if check_amplitude == 4 
            check_amplitude = 0;
            check_zero = 0;
        end
    end
    check_amplitude = 0;
    check_zero = 0;
end
%% Delay generation
POLAR_NRZ_delayed = zeros(500, 700);   % Matrix initialization to store 
                                       % delayed Polar NRZ signals,
                                       % initially filled with zeroes
UNIPOLAR_NRZ_delayed = zeros(500, 700);
POLAR_RZ_delayed = zeros(500, 700);
for i = 1 : realization_num            % loop runs 500 times, once for each realization.
                                       % gets a unique delay.
    % Assign delayed samples to polar NRZ
    POLAR_NRZ_delayed(i, :) = POLAR_NRZ_samples(i, delay(i) + 1 : delay(i) + 700);
    UNIPOLAR_NRZ_delayed(i, :) = UNI_NRZ_samples(i, delay(i) + 1 : delay(i) + 700);
    POLAR_RZ_delayed(i, :) = POLAR_return_to_zero(i, delay(i) + 1 : delay(i) + 700);
end
%% Plot polar_NRZ_delayed for each realization
figure;
for i = 1:4
    subplot(4,1,i);
    plot(t, POLAR_NRZ_delayed(i,:));
    title(['Delayed polar NRZ (Realization ', num2str(i), ')']);
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
end
%% Plot uni_NRZ_delayed for the first 4 realizations
figure;
for i = 1:4
    subplot(4,1,i);
    plot(t, UNIPOLAR_NRZ_delayed(i,:));
    title(['Delayed unipolar NRZ (Realization ', num2str(i), ')']);
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
end
%% Plot delayed Polar RZ for four realizations
figure;
for i = 1:4
    subplot(4,1,i);
    plot(t, POLAR_RZ_delayed(i,:));
    title(['Delayed polar RZ (Realization ', num2str(i), ')']);
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
end
%% Calculating Ensemble Autocorrelation
% Define the starting index for autocorrelation calculation,away from
% random delay
start_index = 351;

% Define the lag range
tau = -350:349;

% Preallocate arrays for autocorrelation results
POLAR_NRZ = zeros(1, 700);           % Polar NRZ autocorrelation preallocation
UNI_NRZ = zeros(1, 700);
POLAR_RZ = zeros(1, 700); 

% Loop through each lag value
for x = tau
    adjusted_x = x + start_index; % Shift the index 
    
    % Loop through each waveform
    for i = 1:500
        % Accumulate the product of delayed waveforms
        POLAR_NRZ(adjusted_x) = POLAR_NRZ(adjusted_x) + POLAR_NRZ_delayed(i, start_index) * POLAR_NRZ_delayed(i, start_index + x);
        UNI_NRZ(adjusted_x) = UNI_NRZ(adjusted_x) + UNIPOLAR_NRZ_delayed(i, start_index) * UNIPOLAR_NRZ_delayed(i, start_index + x);
        POLAR_RZ(adjusted_x) = POLAR_RZ(adjusted_x) + POLAR_RZ_delayed(i, start_index) * POLAR_RZ_delayed(i, start_index + x);
    end
    
    % Average the accumulated values over all waveforms
    POLAR_NRZ(adjusted_x) = POLAR_NRZ(adjusted_x) / 500;
    UNI_NRZ(adjusted_x) = UNI_NRZ(adjusted_x) / 500;
    POLAR_RZ(adjusted_x) = POLAR_RZ(adjusted_x) / 500;
end
R_0 = POLAR_NRZ(start_index); % Index 151 corresponds to tau = 0
fprintf('R(0) across realizations: %.4f\n', R_0);

% Plotting polar NRZ ensemble autocorrelation
figure;
freq1 = -350:349;
plot(freq1, POLAR_NRZ);
title('Polar NRZ Ensemble Autocorrelation'); 
xlabel('Frequency (f)'); 
ylabel('Amplitude'); 

%Plotting unipolar NRZ ensemble autocorrelation
figure;
plot(freq1, UNI_NRZ); 
title('Unipolar NRZ Ensemble Autocorrelation'); 
xlabel('Frequency (f)'); 
ylabel('Amplitude');

% Plotting polar RZ ensemble autocorrelation
figure;
plot(freq1, POLAR_RZ); 
title('Polar RZ Ensemble Autocorrelation'); 
xlabel('Frequency (f)'); 
ylabel('Amplitude'); 
%% Time auto correlation for a single waveform            
POLAR_NRZ_auto_correlation = zeros (1, 300);      % Initialize arrays to store autocorrelation values
UNI_POLAR_NRZ_auto_correlation = zeros (1, 300); 
POLAR_RZ_auto_correlation = zeros (1, 300); 
tau= -150: 149;                                  % Defines lags ranging from -150 to 149.
                                                  % Negative lags: Compare past values with future values.
                                                  % Positive lags: Compare future values with past values.

for x = tau                            % Time shift is the variable
    adjusted_x = x + 151;              % Convert negative lags to positive index 
        for sample = 151:550           % Loop through 400 time samples ,sample position 
                                       % is the variable starting from 1 to 300
        POLAR_NRZ_auto_correlation(adjusted_x) = POLAR_NRZ_auto_correlation(adjusted_x) + POLAR_NRZ_delayed(1, sample) * POLAR_NRZ_delayed(1, sample + x);
        UNI_POLAR_NRZ_auto_correlation(adjusted_x) = UNI_POLAR_NRZ_auto_correlation(adjusted_x) + UNIPOLAR_NRZ_delayed(1,sample)* UNIPOLAR_NRZ_delayed(1,sample+x);
        POLAR_RZ_auto_correlation(adjusted_x) = POLAR_RZ_auto_correlation(adjusted_x) + POLAR_RZ_delayed(1,sample)* POLAR_RZ_delayed(1,sample+x);
        end
   POLAR_NRZ_auto_correlation(adjusted_x) = POLAR_NRZ_auto_correlation(adjusted_x) / 400;   
   UNI_POLAR_NRZ_auto_correlation(adjusted_x) = UNI_POLAR_NRZ_auto_correlation(adjusted_x)/400;
   POLAR_RZ_auto_correlation(adjusted_x) = POLAR_RZ_auto_correlation(adjusted_x)/400;
                                        %Normalizes the correlation by dividing by number of samples.
                                        %This ensures the values are comparable across different waveforms.
end
R_0 = POLAR_NRZ_auto_correlation(151); % Index 151 corresponds to tau = 0
fprintf('R(0) across time: %.4f\n', R_0);

figure;
freq2 = -150:149;
plot(freq2, POLAR_NRZ_auto_correlation);
title('Polar NRZ time autocorrelation');
xlabel('Frequency'); ylabel('Amplitude');

figure;
plot(freq2,UNI_POLAR_NRZ_auto_correlation);
 title('unipolar NRZ time autocorrelation');
 xlabel('f');
 ylabel('Amplitude');

figure;
plot(freq2,POLAR_RZ_auto_correlation);
 title('Polar RZ time autocorrelation');
 xlabel('f');
 ylabel('Amplitude');
%% Statistical Mean across realizations Calculation
% Calculate the statistical mean for polar NRZ
statistical_mean_POLAR_NRZ = zeros(1, size(POLAR_NRZ_delayed, 2)); % A zero array with one row and
                                                                   % the same number of columns
                                                                   % This array will store the 
                                                                   % mean amplitude of the signal
                                                                   % at each time step.
for i = 1:size(POLAR_NRZ_delayed, 2)
    statistical_mean_POLAR_NRZ(i) = sum(POLAR_NRZ_delayed(:, i)) / size(POLAR_NRZ_delayed, 1);
end
    % check 
    mean_across_realizations = mean(POLAR_NRZ_delayed, 1); 
    fprintf('Mean Across Realizations: %.4f\n', mean_across_realizations);

statistical_mean_UNIPOLAR_NRZ = zeros(1, size(UNIPOLAR_NRZ_delayed, 2));
for i = 1:size(UNIPOLAR_NRZ_delayed, 2)
    statistical_mean_UNIPOLAR_NRZ(i) = sum(UNIPOLAR_NRZ_delayed(:, i)) / size(UNIPOLAR_NRZ_delayed, 1);
end

% Calculate the statistical mean for polar RZ
statistical_mean_POLAR_RZ = zeros(1, size(POLAR_RZ_delayed, 2));
for i = 1:size(POLAR_RZ_delayed, 2)
    statistical_mean_POLAR_RZ(i) = sum(POLAR_RZ_delayed(:, i)) / size(POLAR_RZ_delayed, 1);
end

% Plotting the statistical mean 
figure;
plot(t, statistical_mean_POLAR_NRZ);
title('Statistical Mean for Polar NRZ');
xlabel('Time (s)');
ylabel('Amplitude');

figure;
plot(t, statistical_mean_UNIPOLAR_NRZ);
title('Statistical Mean for Unipolar NRZ');
xlabel('Time (s)');
ylabel('Amplitude');

figure;
plot(t, statistical_mean_POLAR_RZ);
title('Statistical Mean for Polar RZ');
xlabel('Time (s)');
ylabel('Amplitude');
%% Calculate the time mean for the selected waveform 
mean_across_time_POLAR_NRZ = sum(POLAR_NRZ_delayed(1, :)) / length(POLAR_NRZ_delayed(1, :));
mean_across_time_UNIPOLAR_NRZ = sum(POLAR_NRZ_delayed(1, :)) / length(POLAR_NRZ_delayed(1, :));
mean_across_time_POLAR_RZ = sum(POLAR_NRZ_delayed(1, :)) / length(POLAR_NRZ_delayed(1, :));
fprintf('Calculated Mean Across Time for a single realization: %.4f\n', mean_across_time_POLAR_NRZ);
% check
mean_across_time = mean(POLAR_NRZ_delayed(1, :)); 


fprintf('Mean Across Time for a single realization: %.4f\n', mean_across_time);
figure;
plot(1:realization_num, repmat(mean_across_time_POLAR_NRZ, realization_num, 1));
title('Mean Across time for Polar NRZ');
xlabel('Realization');
ylabel('Amplitude');

figure;
plot(1:realization_num, repmat(mean_across_time_UNIPOLAR_NRZ, realization_num, 1));
title('Mean Across time for Unipolar NRZ');
xlabel('Realization');
ylabel('Amplitude');

figure;
plot(1:realization_num, repmat(mean_across_time_POLAR_RZ, realization_num, 1));
title('Mean Across time for Polar RZ');
xlabel('Realization');
ylabel('Amplitude');

%% Chat Gpt 
window_size = 50; % Adjust this to smooth variations
smoothed_mean = movmean(mean(POLAR_NRZ_delayed, 1), window_size);

figure;
plot(t, smoothed_mean, 'r', 'LineWidth', 2);
title('Smoothed Statistical Mean for Polar NRZ');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;


check_stationarity_ergodicity(POLAR_NRZ_delayed);
check_stationarity_ergodicity(UNIPOLAR_NRZ_delayed);
check_stationarity_ergodicity(POLAR_RZ_delayed);
function check_stationarity_ergodicity(signal)
    % Function to check if a signal is Stationary and Ergodic
    % signal: Matrix (num_realizations x num_samples)
    
    % Get dimensions
    [num_realizations, num_samples] = size(signal);
    
    % 1️⃣ 📌 Compute Statistical Mean Across Realizations
    ensemble_mean = mean(signal, 1); % Mean at each time step
    
    % 2️⃣ 📌 Compute Variance Over Time Windows (Stationarity Test)
    window_size = 50; % Choose a suitable window size
    num_windows = floor(num_samples / window_size);
    mean_variations = zeros(1, num_windows);
    var_variations = zeros(1, num_windows);
    
    for i = 1:num_windows
        start_idx = (i - 1) * window_size + 1;
        end_idx = start_idx + window_size - 1;
        segment = ensemble_mean(start_idx:end_idx);
        
        mean_variations(i) = mean(segment);
        var_variations(i) = var(segment);
    end
    
    % 📊 Plot Mean and Variance over Time Segments
    figure;
    subplot(2,1,1);
    plot(mean_variations, 'r', 'LineWidth', 2);
    title('Mean Across Time Segments');
    xlabel('Time Segment');
    ylabel('Mean Value');
    
    subplot(2,1,2);
    plot(var_variations, 'b', 'LineWidth', 2);
    title('Variance Across Time Segments');
    xlabel('Time Segment');
    ylabel('Variance');
    
    % 📝 Stationarity Conclusion
    if max(var_variations) - min(var_variations) < 0.1 % Threshold for stationarity
        fprintf('✅ The process is likely Stationary.\n');
    else
        fprintf('❌ The process is NOT Stationary.\n');
    end
    
    % 3️⃣ 📌 Compute Time Mean of a Single Realization
    time_mean = mean(signal(1, :)); % Choose first realization
    
    % 4️⃣ 📌 Compute Mean Across All Realizations
    mean_across_realizations = mean(ensemble_mean);
    
    % 📝 Ergodicity Conclusion
    if abs(time_mean - mean_across_realizations) < 0.1 % Threshold for ergodicity
        fprintf('✅ The process is likely Ergodic.\n');
    else
        fprintf('❌ The process is NOT Ergodic.\n');
    end
end


                                       