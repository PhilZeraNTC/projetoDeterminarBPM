clear; clc; close all;

% Defina o arquivo de áudio
audioFile = 'Harder,Better,Faster,Stronger.mp3'; 

% Carregar e pré-processar o áudio
[y, Fs] = audioread(audioFile); % Carrega o audio, y é uma lista de valores de amplitude do sinal e Fs é a taxa de amostragem em Hz.
if size(y, 2) > 1, y = mean(y, 2); end % Se for um áudio estéro converte em mono
y = y / max(abs(y)); %normaliza y para tratar os valores de amplitude de 0 a 1

% Criar vetor de tempo para o sinal completo
t = (0:length(y)-1) / Fs;

disp(['áudio carregado: ' audioFile]);
disp(['Taxa de amostragem (Fs): ' num2str(Fs) ' Hz']);


% Plots do sinal

disp('Gerando gráficos globais de tempo e frequência...');

figure('Name', 'Análise Global do Sinal');
% Plot no domínio do tempo
subplot(2,1,1);
plot(t, y);
title('Sinal de áudio no Domínio do Tempo');
xlabel('Tempo (s)');
ylabel('Amplitude Normalizada');
grid on;
legend('Sinal de áudio (y)');

% Plot no domínio da frequência (FFT do sinal inteiro)
N = length(y);
Y_fft = fft(y);
P2 = abs(Y_fft/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(N/2))/N;

subplot(2,1,2);
plot(f, P1);
title('Espectro de Frequência Global (FFT)');
xlabel('Frequência (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]);
grid on;
legend('Magnitude do Espectro');


% Análise de Estacionariedade (Média e Variância por Janela)
disp('Analisando estacionariedade com janela deslizante...');

winSize_stats = 2048; % Tamanho da janela para análise estatística
hopSize_stats = 1024; % Salto da janela
num_frames = floor((length(y) - winSize_stats) / hopSize_stats) + 1;

media_movel = zeros(1, num_frames);
variancia_movel = zeros(1, num_frames);
time_stats = zeros(1, num_frames);

for i = 1:num_frames
    startIndex = (i-1)*hopSize_stats + 1;
    endIndex = startIndex + winSize_stats - 1;
    frame = y(startIndex:endIndex);
    
    media_movel(i) = mean(frame);
    variancia_movel(i) = var(frame);
    time_stats(i) = mean(t(startIndex:endIndex)); % Ponto central da janela no tempo
end

figure('Name', 'Análise de Estacionariedade');
subplot(2,1,1);
plot(time_stats, media_movel);
title('Média Móvel do Sinal ao Longo do Tempo');
xlabel('Tempo (s)');
ylabel('Média');
grid on;

subplot(2,1,2);
plot(time_stats, variancia_movel);
title('Variância Móvel do Sinal ao Longo do Tempo');
xlabel('Tempo (s)');
ylabel('Variância');
grid on;


% Visualização Tempo-Frequência (Espectrograma)
disp('Gerando espectrograma para visualizar energia...');

figure('Name', 'Visualização Tempo-Frequência');
spectrogram(y, hann(1024), 512, 1024, Fs, 'yaxis');
title('Espectrograma do Sinal de áudio');
% O espectrograma mostra como a energia (cor) das frequências (eixo Y)
% varia ao longo do tempo (eixo X).


% DETECÇÃO DE BPM VIA FOURIER (STFT)
disp('Iniciando detecção de BPM via STFT...');

% Parâmetros 
frameSize = 1024;
hopSize = frameSize / 2;

% STFT e ODF via Fluxo Espectral
[S, ~, T_stft] = stft(y, Fs, 'Window', hann(frameSize), 'OverlapLength', hopSize, 'FFTLength', frameSize); % Cria frames de tamanho 1024 e faz transformada rápida de fourier nesses quadros para ver a frequencia em cada pequeno instante de tempo
magnitudeSpectrum = abs(S);
spectralFlux = zeros(1, size(magnitudeSpectrum, 2) - 1);
for i = 1:(size(magnitudeSpectrum, 2) - 1)
    spectralFlux(i) = sum(max(0, magnitudeSpectrum(:, i+1) - magnitudeSpectrum(:, i))); % Calcula todas os aumentos de energia do sinal em diferentes frequencias gerando um unico valor por instante.
end
ODF_ft = spectralFlux / max(spectralFlux); % Renomeado para ODF_ft
timeODF_ft = T_stft(1:end-1);

% Suavização e Análise de BPM (FT)
windowSizeSmooth = round(0.1 * Fs / hopSize);
if windowSizeSmooth < 1, windowSizeSmooth = 1; end
ODF_ft_smooth = conv(ODF_ft, ones(1, windowSizeSmooth)/windowSizeSmooth, 'valid'); % Cada ponto da ODF se torna a média de seus vizinhos, evitando pequenas flutuações de timbre ou ruidos.
[bpm_ft, bpm_spectrum_ft, bpm_candidates_ft] = analyze_odf(ODF_ft_smooth, Fs / hopSize);


% DETECÇÃO DE BPM VIA WAVELET (CWT) - MÉTODO COMPARATIVO
disp('Iniciando detecção de BPM via CWT...');

% CWT e ODF via Energia Wavelet 

% SOLUÇÃO PARA O ERRO DE MEMÓRIA:
% Analise de um trecho de 20 segundos do meio da música, onde o ritmo já está bem estabelecido.    
start_time = 30; % Começar em 30 segundos
duration = 20;   % Analisar por 20 segundos
y_cwt = y(start_time*Fs : (start_time+duration)*Fs);        

disp(['Analisando trecho da CWT de ' num2str(start_time) 's a ' num2str(start_time+duration) 's.']);

% Aplicar a CWT apenas no trecho menor
[cfs, f_cwt] = cwt(y_cwt, Fs); % cfs = coeficientes, f = frequências

% A ODF da CWT é a soma da energia (magnitude) dos coeficientes em todas as escalas (frequências) para cada ponto no tempo.
ODF_cwt = sum(abs(cfs), 1);
ODF_cwt = ODF_cwt / max(ODF_cwt);
timeODF_cwt = (0:length(ODF_cwt)-1) / Fs;

% Suavização e Análise de BPM (CWT)
% A taxa de amostragem da ODF da CWT é a mesma do áudio original (Fs)
windowSizeSmooth_cwt = round(0.1 * Fs);
ODF_cwt_smooth = conv(ODF_cwt, ones(1, windowSizeSmooth_cwt)/windowSizeSmooth_cwt, 'valid');
[bpm_cwt, bpm_spectrum_cwt, bpm_candidates_cwt] = analyze_odf(ODF_cwt_smooth, Fs);

% COMPARAÇÃO DOS RESULTADOS
disp('Gerando gráficos comparativos...');

minBPM = 50;
maxBPM = 250;

% Comparação das ODFs
figure('Name', 'Comparativo das ODFs');
subplot(2,1,1);
plot(timeODF_ft(1:length(ODF_ft_smooth)), ODF_ft_smooth, 'b'); 
title('ODF Suavizada a partir da STFT (Fluxo Espectral)');
xlabel('Tempo (s)'); ylabel('Amplitude Normalizada'); grid on; xlim([5 15]);

subplot(2,1,2);
plot(timeODF_cwt(1:length(ODF_cwt_smooth)), ODF_cwt_smooth, 'g');
title('ODF Suavizada a partir da CWT (Energia Wavelet)');
xlabel('Tempo (s)'); ylabel('Amplitude Normalizada'); grid on; xlim([5 15]);

% Comparação dos Espectros de BPM
figure('Name', 'Comparativo dos Espectros de BPM');
hold on;
% CORREÇÃO 1: Plotar a partir do segundo elemento para ignorar o Componente DC
plot(bpm_candidates_ft(2:end), bpm_spectrum_ft(2:end), 'b', 'DisplayName', 'Método STFT');
plot(bpm_candidates_cwt(2:end), bpm_spectrum_cwt(2:end), 'g', 'LineWidth', 1.5, 'DisplayName', 'Método CWT');

if ~isnan(bpm_ft)
    xline(bpm_ft, '--b', ['BPM (STFT): ' num2str(bpm_ft, '%.1f')]);
end
if ~isnan(bpm_cwt)
    xline(bpm_cwt, '--g', ['BPM (CWT): ' num2str(bpm_cwt, '%.1f')]);
end

title('Comparativo dos Espectros de BPM: STFT vs. CWT');
xlabel('BPM');
ylabel('Magnitude');
grid on;
legend;

% CORREÇÃO 2: Limitar a visualização do eixo X para a faixa de interesse
xlim([minBPM-10 maxBPM+10]);

hold off;

fprintf('BPM Estimado (STFT): %.2f\n', bpm_ft);
fprintf('BPM Estimado (CWT): %.2f\n', bpm_cwt);


% FUNÇÃO AUXILIAR DE ANÁLISE DE ODF
function [estimatedBPM, Y_magnitude, bpm_candidates] = analyze_odf(odf, Fs_odf)
    % Esta função encapsula a análise de FFT de uma ODF para evitar repetição de código
    N = length(odf);
    Y_fft = fft(odf);
    
    frequencies = (0:N-1) * (Fs_odf / N);
    Y_magnitude = abs(Y_fft(1:floor(N/2)+1));
    frequencies_half = frequencies(1:floor(N/2)+1);
    
    bpm_candidates = frequencies_half * 60;
    
    minBPM = 50; maxBPM = 250;
    valid_indices = find(bpm_candidates >= minBPM & bpm_candidates <= maxBPM);
    valid_magnitudes = Y_magnitude(valid_indices);
    valid_bpms = bpm_candidates(valid_indices);
    
    if isempty(valid_magnitudes)
        estimatedBPM = NaN;
    else
        [~, max_idx] = max(valid_magnitudes);
        estimatedBPM = valid_bpms(max_idx);
    end
end
