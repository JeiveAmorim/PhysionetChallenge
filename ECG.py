import mne
import os
import matplotlib.pyplot as plt

# Caminho para o primeiro arquivo baixado (ajuste conforme sua pasta data/raw)
path_to_edf = 'data/raw/training_set/physiological_data/S0001/sub-S0001111197789-ses1.edf'

try:
    # 1. Carregar o arquivo EDF
    # preload=False economiza memória para arquivos grandes de polissonografia
    raw = mne.io.read_raw_edf(path_to_edf, preload=True)
    
    # 2. Informações Básicas
    print(f"\n--- Informações do Exame ---")
    print(f"Canais encontrados: {raw.ch_names}")
    print(f"Frequência de amostragem: {raw.info['sfreq']} Hz")
    print(f"Duração total: {raw.times[-1] / 3600:.2f} horas")

    # 3. Filtragem Simples para Limpeza de Sinal (Opcional para visualização)
    # Filtro passa-banda comum para EEG (0.5 a 30 Hz)
    raw_check = raw.copy().filter(l_freq=0.5, h_freq=30.0, picks='eeg')

    # 4. Plotar um trecho de 20 segundos para verificar a qualidade
    # Vamos focar nos primeiros canais de EEG
    raw_check.plot(duration=20, n_channels=5, scalings='auto', title='Check de Qualidade - EEG')
    
    # 5. Visualizar a Densidade Espectral (PSD)
    # Fundamental para ver se há ruído de rede elétrica (60Hz) ou se as bandas alpha/delta estão visíveis
    raw_check.compute_psd(fmax=50).plot()
    plt.show()

except FileNotFoundError:
    print(f"Arquivo não encontrado em: {path_to_edf}. Verifique se o download da API já criou as pastas.")
except Exception as e:
    print(f"Erro ao ler o sinal: {e}")