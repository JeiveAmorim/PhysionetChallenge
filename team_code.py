#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
import sys
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from scipy.signal import welch
import antropy as ant

import yasa
import logging

# Silencia os avisos chatos do YASA, mostrando apenas Erros críticos
logging.getLogger('yasa').setLevel(logging.ERROR)

from helper_code import *

################################################################################
# Path & Constant Configuration (Added for Robustness)
################################################################################

# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file relative to the script location
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    if verbose:
        print('Encontrando os dados do Challenge...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('Nenhum dado foi fornecido.')

    features = []
    labels = []
    
    pbar = tqdm(range(num_records), desc="Extraindo Features", unit="paciente", disable=not verbose)
    for i in pbar:
        try:
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id    = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            # 1. Demografia
            patient_data = load_demographics(patient_data_file, patient_id, session_id)
            demographic_features = extract_demographic_features(patient_data)

            # 2. Anotações Algorítmicas
            algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
            if os.path.exists(algo_file):
                algo_data, algo_fs = load_signal_data(algo_file)
                algorithmic_features = extract_algorithmic_annotations_features(algo_data)
            else:
                algo_data, algo_fs = None, {}
                algorithmic_features = np.zeros(12)

            # 3. Fisiologia Multimodal (O seu vetor de ~48 colunas sem o EEG)
            phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
            if os.path.exists(phys_file):
                phys_data, phys_fs = load_signal_data(phys_file)
                physiological_features = extract_physiological_features(phys_data, phys_fs, algo_data, algo_fs, csv_path=csv_path)
            else:
                # O fallback agora deve ter o mesmo tamanho da sua função sem EEG (ajuste se for diferente de 48)
                physiological_features = np.zeros(48) 

            # 4. Diagnóstico (Label)
            label = load_diagnoses(patient_data_file, patient_id)

            # 5. Concatenação Completa
            if label == 0 or label == 1:
                features.append(np.hstack([demographic_features, physiological_features, algorithmic_features]))
                labels.append(label)

            if 'algo_data' in locals(): del algo_data
            if 'phys_data' in locals(): del phys_data

        except Exception as e:
            tqdm.write(f"  !!! Erro ao processar registro {i+1} ({patient_id}): {e}")
            continue

    pbar.close()

    # === TREINAMENTO ===
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)

    if verbose:
        print(f'\nTamanho total extraído: {features.shape}')
        print('Treinando a Random Forest definitiva com os hiperparâmetros otimizados...')

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='log2',
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(features, labels)

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model)

    if verbose:
        print('Treinamento concluído e modelo salvo com sucesso!')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(model, record, data_folder, verbose):
    # Load the model.
    model = model['model']

    # Extract identifiers from the record dictionary
    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # Load the patient data.
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(patient_data_file, patient_id, session_id)
    demographic_features = extract_demographic_features(patient_data)

    # Load signal data.
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(phys_file):
        phys_data, phys_fs = load_signal_data(phys_file)
        # Ensure csv_path is accessible or defined
        physiological_features = extract_physiological_features(phys_data, phys_fs)
    else:
        # Fallback to zeros if file is missing (length 49)
        physiological_features = np.zeros(49)

    # Load Algorithmic Annotations
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        algorithmic_features = extract_algorithmic_annotations_features(algo_data)
    else:
        # Fallback to zeros (length 12)
        algorithmic_features = np.zeros(12)

    features = np.hstack([demographic_features, physiological_features, algorithmic_features]).reshape(1, -1)

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


def spectral_features(signal, fs):

    BANDS = {
        "delta": (0.5,4),
        "theta": (4,8),
        "alpha": (8,13),
        "beta": (13,30),
        "gamma": (30,40)
    }

    freqs, psd = welch(
        signal,
        fs=fs,
        window='hamming',
        nperseg=5*fs,
        noverlap=int(2.5*fs)
    )

    mask = (freqs >= 0.5) & (freqs <= 40)
    
    total_power = np.trapezoid(psd[mask], freqs[mask])
    total_power = max(total_power, 1e-10)

    features = {}

    for band in BANDS:
        low, high = BANDS[band]
        idx = (freqs >= low) & (freqs <= high)

        band_power = np.trapezoid(psd[idx], freqs[idx])

        features[f"{band}_abs_power"] = band_power
        features[f"{band}_rel_power"] = band_power / total_power

    features["total_power"] = total_power

    return features

def complexity_features(signal):

    features = {}

    try:
        features["sample_entropy"] = ant.sample_entropy(signal)
    except:
        features["sample_entropy"] = np.nan

    # app_entropy REMOVIDA AQUI para ganho extremo de performance (Otimização Nível 1)

    try:
        features["higuchi_fd"] = ant.higuchi_fd(signal)
    except:
        features["higuchi_fd"] = np.nan

    return features

def spindle_features(signal, fs):

    features = {
        "spindle_count": 0,
        "spindle_duration_mean": 0,
        "spindle_amp_mean": 0,
        "spindle_freq_mean": 0
    }

    try:
        sp = yasa.spindles_detect(signal, sf=fs)

        if sp is None:
            return features

        df = sp.summary()

        if df.empty:
            return features

        features["spindle_count"] = len(df)
        features["spindle_duration_mean"] = df["Duration"].mean()
        features["spindle_amp_mean"] = df["Amplitude"].mean()
        features["spindle_freq_mean"] = df["Frequency"].mean()

    except:
        pass

    return features

def slowwave_features(signal, fs):

    features = {
        "sw_count": 0,
        "sw_amp_mean": 0,
        "sw_duration_mean": 0,
        "sw_slope_mean": 0
    }

    try:
        sw = yasa.sw_detect(signal, sf=fs)

        if sw is None:
            return features

        df = sw.summary()

        if df.empty:
            return features

        features["sw_count"] = len(df)
        features["sw_amp_mean"] = df["PTP"].mean()
        features["sw_duration_mean"] = df["Duration"].mean()
        features["sw_slope_mean"] = df["Slope"].mean()

    except:
        pass

    return features

def extract_epoch_features(epoch_signal, fs, stage):

    feat = {}

    # espectral + complexidade sempre
    feat.update(spectral_features(epoch_signal, fs))
    feat.update(complexity_features(epoch_signal))

    # spindles apenas N2
    if stage == "N2":
        feat.update(spindle_features(epoch_signal, fs))
    else:
        feat.update({
            "spindle_count": 0,
            "spindle_duration_mean": 0,
            "spindle_amp_mean": 0,
            "spindle_freq_mean": 0
        })

    # slow waves apenas N3
    if stage == "N3":
        feat.update(slowwave_features(epoch_signal, fs))
    else:
        feat.update({
            "sw_count": 0,
            "sw_amp_mean": 0,
            "sw_duration_mean": 0,
            "sw_slope_mean": 0
        })

    return feat

def extract_eeg_features(sig, fs, canais, hypnoIn, epoch_sec=30):
    """
    Versão Otimizada (Sem app_entropy). 
    Garante saída de tamanho fixo: 4 canais * 5 estágios * 21 features = 420 posições.
    """
    EXPECTED_CHANNELS = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1']
    SLEEP_STAGES = ["W", "N1", "N2", "N3", "REM"]
    
    # REMOVIDO: "app_entropy" (Agora são 21 features por época)
    FEAT_KEYS = [
        "delta_abs_power", "delta_rel_power", "theta_abs_power", "theta_rel_power",
        "alpha_abs_power", "alpha_rel_power", "beta_abs_power", "beta_rel_power",
        "gamma_abs_power", "gamma_rel_power", "total_power",
        "sample_entropy", "higuchi_fd",
        "spindle_count", "spindle_duration_mean", "spindle_amp_mean", "spindle_freq_mean",
        "sw_count", "sw_amp_mean", "sw_duration_mean", "sw_slope_mean"
    ]

    mapeamento = {1: 'W', 2: 'N1', 3: 'N2', 4: 'N3', 5: 'REM', 9: 'NC'}
    hypno = np.vectorize(mapeamento.get)(hypnoIn)

    data_map = {c.lower(): (s, f) for s, f, c in zip(sig, fs, canais)}
    final_vector = []
    
    for ch_name in EXPECTED_CHANNELS:
        if ch_name in data_map:
            signal_data, fs_ch = data_map[ch_name]
            epoch_len = int(epoch_sec * fs_ch)
            
            stage_epochs = {s: [] for s in SLEEP_STAGES}
            for i in range(min(len(hypno), len(signal_data) // epoch_len)):
                start = i * epoch_len
                end = start + epoch_len
                st = hypno[i]
                if st in SLEEP_STAGES:
                    stage_epochs[st].append(signal_data[start:end])
            
            for st in SLEEP_STAGES:
                epochs = stage_epochs[st]
                if len(epochs) > 0:
                    epoch_results = [extract_epoch_features(ep, fs_ch, st) for ep in epochs]
                    for k in FEAT_KEYS:
                        vals = [res.get(k, 0.0) for res in epoch_results]
                        val_medio = np.nan_to_num(np.nanmean(vals), nan=0.0)
                        final_vector.append(val_medio)
                else:
                    # ATUALIZADO: 21 zeros (estágio ausente)
                    final_vector.extend([0.0] * 21) 
        else:
            # ATUALIZADO: 105 zeros (canal ausente = 5 estágios * 21 métricas)
            final_vector.extend([0.0] * 105) 
            
    return np.array(final_vector).reshape(1, -1)

def extract_hrv_features(sig, fs, sleep_stages, sleep_stages_fs):
    """
    Extrai features de tempo e frequência (HRV) para cada estágio do sono.
    Mapeamento: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake.
    Retorna um vetor fixo de 25 posições (5 features x 5 estágios).
    """
    # Se não tiver dados suficientes, retorna o vetor zerado (25 features)
    if sig is None or len(sig) < fs * 60 or len(sleep_stages) == 0:
        return [0.0] * 25

    try:
        # 1. Detecção de Picos R super rápida (Pan-Tompkins)
        peaks_info = nk.ecg_peaks(sig, sampling_rate=fs, method="pantompkins1985")[1]
        r_peaks = peaks_info["ECG_R_Peaks"]
        
        if len(r_peaks) < 10:
            return [0.0] * 25

        # 2. Calcular os Intervalos RR e seus Timestamps
        rr_times = r_peaks[:-1] / fs  # Em segundos
        rr_intervals = np.diff(r_peaks) * (1000.0 / fs)  # Em milissegundos

        # 3. Filtro de Sanidade (Remove ruídos de movimento que quebram o HRV)
        valid_mask = (rr_intervals > 300) & (rr_intervals < 2000)
        rr_times = rr_times[valid_mask]
        rr_intervals = rr_intervals[valid_mask]

        # 4. Encontrar o estágio do sono para cada intervalo RR
        # Época = tempo * frequência_de_amostragem_das_anotações
        stage_indices = np.floor(rr_times * sleep_stages_fs).astype(int)
        
        # Garante que não ultrapassamos o tamanho do array de estágios
        stage_indices = np.clip(stage_indices, 0, len(sleep_stages) - 1)
        rr_stages = sleep_stages[stage_indices]

        # 5. Extração de Features Agrupadas por Estágio
        features = []
        target_stages = [1, 2, 3, 4, 5] # N3, N2, N1, REM, Wake

        for stage in target_stages:
            mask = (rr_stages == stage)
            stage_rrs = rr_intervals[mask]
            stage_times = rr_times[mask]

            # Se houver pelo menos 20 batimentos válidos neste estágio
            if len(stage_rrs) > 20:
                # --- Domínio do Tempo ---
                sdnn = np.std(stage_rrs, ddof=1)
                rmssd = np.sqrt(np.mean(np.square(np.diff(stage_rrs))))
                
                # --- Domínio da Frequência (Requer Interpolação) ---
                # Garante que não há tempos duplicados para a interpolação funcionar
                _, unique_idx = np.unique(stage_times, return_index=True)
                t_clean = stage_times[unique_idx]
                rr_clean = stage_rrs[unique_idx]

                if len(t_clean) > 20:
                    # Interpola para 4Hz (padrão ouro em HRV)
                    fs_interp = 4.0
                    t_interp = np.arange(t_clean[0], t_clean[-1], 1.0/fs_interp)
                    f_interp = scipy.interpolate.interp1d(t_clean, rr_clean, kind='cubic', fill_value='extrapolate')
                    rr_interp = f_interp(t_interp)

                    # Welch's Periodogram
                    f, pxx = signal.welch(rr_interp, fs=fs_interp, nperseg=min(256, len(rr_interp)))
                    
                    df = f[1] - f[0]
                    lf = np.sum(pxx[(f >= 0.04) & (f <= 0.15)]) * df
                    hf = np.sum(pxx[(f >= 0.15) & (f <= 0.40)]) * df
                    lf_hf = lf / hf if hf > 0 else 0.0
                else:
                    lf, hf, lf_hf = 0.0, 0.0, 0.0

                features.extend([rmssd, sdnn, lf, hf, lf_hf])
            else:
                # Paciente não teve esse estágio do sono ou sinal estava muito ruim
                features.extend([np.nan] * 5)
        
        return features

    except Exception as e:
        # Silencia erros para não quebrar o script de avaliação no servidor
        return [np.nan] * 25

def limpar_sinal_resp(sinal, fs, low_freq, high_freq):
    """
    Filtro passa-banda adaptado do membro da equipe para limpar sinais respiratórios.
    """
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)
    if low < high:
        # Usa o scipy.signal que já está importado como 'signal' no seu topo
        sos = signal.butter(4, [low, high], btype='bandpass', output='sos')
        return signal.sosfiltfilt(sos, sinal)
    return np.asarray(sinal)

def extract_respiratory_features(sig, fs, canais):
    """ 
    Extrai features respiratórias (O2, Ronco, Esforço e Fluxo).
    Espera receber apenas os canais do tipo 'resp' e 'spo2' agrupados.
    Retorna um array com 6 posições.
    """
    # Inicializamos com 0.0 (em vez de NaN) para evitar quebrar a RandomForest
    features_resp = [0.0] * 6 
    
    if not sig or len(sig) == 0:
        return features_resp

    try:
        def find_col_idx(keys): 
            return next((i for i, c in enumerate(canais) if any(k in c.lower() for k in keys)), None)

        idx_o2 = find_col_idx(['spo2', 'sao2', 'osat', 'o2sat'])
        idx_fluxo = find_col_idx(['ptaf', 'pressure', 'airflow', 'cflow', 'nasal', 'thermistor'])
        idx_chest = find_col_idx(['chest', 'thorax', 'thoracic'])
        idx_abd = find_col_idx(['abd', 'abdomen'])
        
        def obter_fs(idx):
            return fs[idx] if isinstance(fs, (list, tuple, np.ndarray)) else fs

        feat_dict = {
            'T90_Pct': 0.0, 'Carga_Hipoxica': 0.0, 'Instabilidade_O2': 0.0,    
            'Ronco_Pct': 0.0, 'Var_Fluxo_Resp': 0.0, 'Assincronia_Torax_Abd': 0.0
        }

        # --- BLOCO 1: SANGUE E OXIGÉNIO ---
        if idx_o2 is not None:
            sinal_o2 = np.asarray(sig[idx_o2])
            fs_o2 = obter_fs(idx_o2)
            
            sinal_o2 = np.clip(sinal_o2, a_min=None, a_max=100.0)
            if sinal_o2.max() <= 1.0: 
                sinal_o2 *= 100.0
            
            feat_dict['T90_Pct'] = ((sinal_o2 < 90.0).sum() / len(sinal_o2)) * 100.0
            feat_dict['Carga_Hipoxica'] = np.maximum(0, 90.0 - sinal_o2).sum() / fs_o2
            feat_dict['Instabilidade_O2'] = np.std(np.diff(sinal_o2))

        # --- BLOCO 2: RONCO E FLUXO ---
        if idx_fluxo is not None:
            sinal_bruto = np.asarray(sig[idx_fluxo])
            fs_fluxo = obter_fs(idx_fluxo)
            
            # Ronco (> 15 Hz)
            sinal_ronco = limpar_sinal_resp(sinal_bruto, fs_fluxo, low_freq=15.0, high_freq=fs_fluxo/2.1)
            energia_ronco = pd.Series(sinal_ronco).abs().rolling(int(fs_fluxo), center=True).mean().bfill().ffill()
            limiar_ronco = energia_ronco.mean() + (1.5 * energia_ronco.std())
            feat_dict['Ronco_Pct'] = ((energia_ronco > limiar_ronco).sum() / len(energia_ronco)) * 100.0
            
            # Variabilidade da Respiração (0.1 a 3 Hz)
            sinal_resp = limpar_sinal_resp(sinal_bruto, fs_fluxo, low_freq=0.1, high_freq=3.0)
            envelope_resp = pd.Series(sinal_resp).abs().rolling(int(fs_fluxo * 2)).mean()
            if envelope_resp.mean() > 0:
                feat_dict['Var_Fluxo_Resp'] = envelope_resp.std() / envelope_resp.mean()

        # --- BLOCO 3: ESFORÇO E MECÂNICA ---
        if idx_chest is not None and idx_abd is not None:
            sinal_c = np.asarray(sig[idx_chest])
            fs_c = obter_fs(idx_chest)
            
            sinal_a = np.asarray(sig[idx_abd])
            fs_a = obter_fs(idx_abd)
            
            sinal_c_limpo = limpar_sinal_resp(sinal_c, fs_c, low_freq=0.1, high_freq=1.0)
            sinal_a_limpo = limpar_sinal_resp(sinal_a, fs_a, low_freq=0.1, high_freq=1.0)
            
            min_len = min(len(sinal_c_limpo), len(sinal_a_limpo))
            if min_len > 1:
                correlacao = np.corrcoef(sinal_c_limpo[:min_len], sinal_a_limpo[:min_len])[0, 1]
                feat_dict['Assincronia_Torax_Abd'] = 1.0 - correlacao

        # Monta a lista final na mesma ordem
        features_resp = [
            feat_dict['T90_Pct'], feat_dict['Carga_Hipoxica'], feat_dict['Instabilidade_O2'],
            feat_dict['Ronco_Pct'], feat_dict['Var_Fluxo_Resp'], feat_dict['Assincronia_Torax_Abd']
        ]

    except Exception as e:
        # Pass silencia o erro e retorna os zeros
        pass
        
    # Garante que não sobrou nenhum NaN que possa quebrar a RandomForest
    features_resp = np.nan_to_num(features_resp, nan=0.0).tolist()
    return features_resp

def extract_eog_features(sig, fs):
    """
    Extrai features específicas do EOG (Movimento dos olhos).
    Espera receber uma lista com 2 sinais de EOG (ex: E1 e E2).
    Retorna exatamente 3 features: [cross_corr, movement_density, spectral_edge_95].
    """
    default_out = [0.0, 0.0, 0.0]
    
    # Precisamos de exatamente 2 canais para cruzar as informações
    if not sig or len(sig) < 2:
        return default_out
        
    try:
        eog1 = np.asarray(sig[0], dtype=np.float64).squeeze()
        eog2 = np.asarray(sig[1], dtype=np.float64).squeeze()
        fs_used = float(fs[0])
        
        if fs_used <= 0 or eog1.ndim != 1 or eog2.ndim != 1 or len(eog1) < 2 or len(eog2) < 2:
            return default_out
            
        # Iguala o tamanho dos dois sinais caso haja diferença de amostras
        min_len = min(len(eog1), len(eog2))
        eog1, eog2 = eog1[:min_len], eog2[:min_len]
        combined = eog1 - eog2 # A diferença destaca o movimento conjugado dos olhos
        
        features = [0.0, 0.0, 0.0]
        
        # 1) Cross-correlation (Muito negativa no REM)
        cross_corr = np.corrcoef(eog1, eog2)[0, 1]
        if np.isfinite(cross_corr):
            features[0] = cross_corr
            
        # 2) Densidade de Movimento
        diff_sig = np.abs(np.diff(combined))
        if len(diff_sig) > 0:
            threshold = np.percentile(diff_sig, 90)
            mask = (diff_sig >= threshold).astype(int)
            events = np.sum((mask[1:] == 1) & (mask[:-1] == 0))
            duration_min = len(combined) / fs_used / 60.0
            if duration_min > 0:
                features[1] = events / duration_min
                
        # 3) Spectral Edge 95%
        nperseg = min(len(combined), int(5 * fs_used))
        if nperseg >= 8:
            # Usando a biblioteca signal que já está importada
            freqs, psd = signal.welch(combined, fs=fs_used, window="hamming", nperseg=nperseg, noverlap=nperseg // 2)
            mask_f = (freqs >= 0.1) & (freqs <= 15)
            if np.any(mask_f):
                psd_sel = psd[mask_f]
                freqs_sel = freqs[mask_f]
                if np.sum(psd_sel) > 0:
                    cumulative = np.cumsum(psd_sel)
                    idx = np.searchsorted(cumulative, 0.95 * cumulative[-1])
                    features[2] = freqs_sel[min(idx, len(freqs_sel) - 1)]
                    
        return features
        
    except Exception as e:
        return default_out

def extract_demographic_features(data):
    """
    Extracts and encodes demographic features from a metadata dictionary.
    
    Inputs:
        data (dict): A dictionary containing patient metadata (e.g., from a CSV row).
    
    Returns:
        np.array: A feature vector of length 11:
            - [0]: Age (Continuous)
            - [1:4]: Sex (One-hot: Female, Male, Other/Unknown)
            - [4:9]: Race (One-hot: Asian, Black, Other, Unavailable, White)
            - [9]: BMI (Continuous)
    """
    # 1. Age Feature (1 dimension)
    # Convert 'Age' to a float; default to 0 if missing
    age = np.array([load_age(data)])

    # 2. Sex One-Hot Encoding (3 dimensions: Female, Male, Other/Unknown)
    # Uses lowercase prefix matching to handle variants like 'F', 'Female', 'M', or 'Male'
    sex = load_sex(data)
    sex_vec = np.zeros(3)
    if sex == 'Female': 
        sex_vec[0] = 1 # Index 0: Female
    elif sex == 'Male': 
        sex_vec[1] = 1 # Index 1: Male
    else: 
        sex_vec[2] = 1 # Index 2: Other/Unknown

    # 3. Race One-Hot Encoding (6 dimensions)
    # Standardizes the raw text into one of six categories using the helper function
    race_category = get_standardized_race(data).lower()
    race_vec = np.zeros(5)
    # Pre-defined mapping for index consistency
    race_mapping = {'asian': 0, 'black': 1, 'others': 2, 'unavailable': 3, 'white': 4}
    race_vec[race_mapping.get(race_category, 2)] = 1

    # 4. Body Mass Index (BMI) Feature (1 dimension)
    # Extracts the pre-calculated mean BMI; handles strings, NaNs, and missing keys
    bmi = np.array([load_bmi(data)])

    # 5. Concatenate all components into a single vector (1 + 3 + 5 + 1 = 10)
    
    return np.concatenate([age, sex_vec, race_vec, bmi])

def extract_physiological_features(physiological_data, physiological_fs, algorithmic_annotations, algorithmic_fs, csv_path=DEFAULT_CSV_PATH):
    """
    Standardizes channels and extracts multimodal physiological features.
    Garante a saída de um vetor de tamanho fixo (488 colunas).
    """
    # --- 1. Extração Segura dos Estágios do Sono ---
    if algorithmic_annotations is not None and 'stage_caisr' in algorithmic_annotations:
        sleep_stages = algorithmic_annotations['stage_caisr']
        sleep_stages_fs = algorithmic_fs.get('stage_caisr', 1/30.0)
    else:
        sleep_stages = np.array([])
        sleep_stages_fs = 1/30.0

    original_labels = list(physiological_data.keys())
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop: continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        processed_fs[new_label] = physiological_fs.get(old_label, 200.0) 
    
    if 'physiological_data' in locals(): del physiological_data

    # --- 2. Derivações Bipolares ---
    bipolar_configs = [
        ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
        ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
        ('chin1-chin2', 'chin1', ['chin2']),
        ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-'])
    ]
    for target, pos, neg_list in bipolar_configs:
        if target in processed_channels or pos not in processed_channels: continue
        if not all(n in processed_channels for n in neg_list): continue
        fs_values = [processed_fs[ch] for ch in [pos] + neg_list]
        if len(set(fs_values)) > 1: continue

        ref_sig = processed_channels[neg_list[0]] if len(neg_list) == 1 else tuple(processed_channels[n] for n in neg_list)
        derived = derive_bipolar_signal(processed_channels[pos], ref_sig)
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target] = processed_fs[pos]

    # --- 3. Extração Multimodal de Features ---
    final_features = []

    # # A) EEG (Opção B - A função garante 440 colunas)
    # eeg_cands = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1']
    # eeg_sigs, eeg_fss, eeg_names = [], [], []
    # for cand in eeg_cands:
    #     if cand in processed_channels:
    #         eeg_sigs.append(processed_channels[cand])
    #         eeg_fss.append(processed_fs[cand])
    #         eeg_names.append(cand)
    
    # # Passamos todas as listas e damos o flatten para transformar a matriz em vetor 1D
    # eeg_vec = extract_eeg_features(eeg_sigs, eeg_fss, eeg_names, sleep_stages)
    # final_features.extend(eeg_vec.flatten().tolist())

    # B) ECG (Sempre 25 colunas)
    ecg_found = False
    for cand in ['ecg', 'ekg']:
        if cand in processed_channels:
            hrv_feats = extract_hrv_features(processed_channels[cand], processed_fs[cand], sleep_stages, sleep_stages_fs)
            final_features.extend(hrv_feats)
            ecg_found = True
            break
    if not ecg_found: 
        final_features.extend([0.0] * 25)

    # C) Respiratório & SpO2 (Sempre 6 colunas)
    resp_cands = ['airflow', 'ptaf', 'abd', 'chest', 'spo2', 'sao2', 'osat', 'o2sat']
    r_sigs, r_fss, r_names = [], [], []
    for cand in resp_cands:
        if cand in processed_channels:
            r_sigs.append(processed_channels[cand])
            r_fss.append(processed_fs[cand])
            r_names.append(cand)
    
    if r_sigs:
        resp_vec = extract_respiratory_features(r_sigs, r_fss, r_names)
        final_features.extend(resp_vec)
    else: 
        final_features.extend([0.0] * 6)

    # D) EOG (Sempre 3 colunas)
    eog_cands = ['e1-m2', 'e2-m1', 'e1', 'e2']
    eog_sigs, eog_fss = [], []
    for cand in eog_cands:
        if cand in processed_channels:
            eog_sigs.append(processed_channels[cand])
            eog_fss.append(processed_fs[cand])
    
    if len(eog_sigs) >= 2:
        eog_vec = extract_eog_features(eog_sigs[:2], eog_fss[:2])
        final_features.extend(eog_vec)
    else:
        final_features.extend([0.0] * 3)

    # E) Outros (Chin, Leg - Sempre 7 colunas cada = 14 total)
    # for lead_type in ['chin', 'leg']:
    #     cands = {'chin': ['chin1-chin2', 'chin'], 'leg': ['lat', 'rat']}[lead_type]
    #     sig_found = False
    #     for cand in cands:
    #         if cand in processed_channels:
    #             s, f = processed_channels[cand], processed_fs[cand]
    #             # Extrai estatísticas e Hjorth parameters
    #             stats = [np.std(s), np.mean(np.abs(s)), np.mean(np.diff(np.sign(s)) != 0), np.sqrt(np.mean(s**2)), np.var(s)]
    #             diff_s = np.diff(s)
    #             mob = np.sqrt(np.var(diff_s)/np.var(s)) if np.var(s) > 0 else 0.0
    #             comp = (np.sqrt(np.var(np.diff(diff_s))/np.var(diff_s))/mob) if (np.var(diff_s) > 0 and mob > 0) else 0.0
                
    #             final_features.extend(stats + [mob, comp])
    #             sig_found = True
    #             break
    #     if not sig_found: 
    #         final_features.extend([0.0] * 7)

    if 'processed_channels' in locals(): del processed_channels
    return np.array(final_features)

def extract_algorithmic_annotations_features(algo_data):
    """
    Extracts sleep architecture and event density features from CAISR outputs.
    Output vector length: 12
    """
    if not algo_data:
        return np.zeros(12)

    features = []

    # --- 1. Respiratory & Arousal Event Densities ---
    # Total duration in hours (assuming 1Hz for event traces)
    # If the signal exists, we calculate events per hour (Index)
    total_hours = len(algo_data.get('resp_caisr', [])) / 3600.0
    
    def count_discrete_events(key):
        if key not in algo_data or total_hours <= 0:
            return 0.0
        
        sig = algo_data[key].astype(float)
        # Create a binary mask: 1 if there is an event, 0 if not
        binary_sig = (sig > 0).astype(int)
        
        # Detect rising edges: 0 to 1 transition
        # diff will be 1 at the start of an event, -1 at the end
        diff = np.diff(binary_sig, prepend=0)
        num_events = np.count_nonzero(diff == 1)
        
        return num_events / total_hours
    
    ahi_auto = count_discrete_events('resp_caisr')      # Automated Apnea-Hypopnea Index
    arousal_auto = count_discrete_events('arousal_caisr') # Automated Arousal Index
    limb_auto = count_discrete_events('limb_caisr')    # Automated Limb Movement Index
    
    features.extend([ahi_auto, arousal_auto, limb_auto])

    # --- 2. Sleep Architecture (from stage_caisr) ---
    # Standard labels: 0=W, 1=N1, 2=N2, 3=N3, 4=R (or similar mapping)
    stages = algo_data.get('stage_caisr', np.array([]))
    # Filter out invalid/background values (like the 9.0 in your sample)
    valid_stages = stages[stages < 9.0]
    
    if len(valid_stages) > 0:
        total_epochs = len(valid_stages)
        # Percentage of each stage
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        
        # Sleep Efficiency: (N1+N2+N3+R) / Total
        efficiency = np.mean(valid_stages > 0)
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = 0.0

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Model Confidence / Uncertainty ---
    # Mean probability of Wake and REM (indicators of sleep stability)
    # We use the raw probability traces
    prob_w = np.mean(algo_data.get('caisr_prob_w', [0]))
    prob_n3 = np.mean(algo_data.get('caisr_prob_n3', [0]))
    prob_arous = np.mean(algo_data.get('caisr_prob_arous', [0]))
    
    # Standardize '9.0' or other filler values to 0
    clean_prob = lambda x: x if x < 1.0 else 0.0
    features.extend([clean_prob(prob_w), clean_prob(prob_n3), clean_prob(prob_arous)])

    return np.array(features)

def extract_human_annotations_features(human_data):
    """
    Extracts features from expert-scored human annotations.
    Output vector length: 12 (to match algorithmic feature length)
    """
    # If data is missing (common in hidden test sets), return a zero vector
    if not human_data or 'resp_expert' not in human_data:
        return np.zeros(12)

    features = []

    # --- 1. Human Event Indices (Events per Hour) ---
    # Total duration in hours based on 1Hz signal
    total_seconds = len(human_data.get('resp_expert', []))
    total_hours = total_seconds / 3600.0
    
    def count_discrete_events(key):
        if key not in human_data or total_hours <= 0:
            return 0.0
        sig = (human_data[key] > 0).astype(int)
        # Identify the start of each continuous event block
        diff = np.diff(sig, prepend=0)
        return np.count_nonzero(diff == 1) / total_hours

    ahi_human = count_discrete_events('resp_expert')      # Human AHI
    arousal_human = count_discrete_events('arousal_expert') # Human Arousal Index
    limb_human = count_discrete_events('limb_expert')       # Human PLMI
    
    features.extend([ahi_human, arousal_human, limb_human])

    # --- 2. Human Sleep Architecture ---
    # Standard labels: 0=W, 1=N1, 2=N2, 3=N3, 4=R, 5=Unknown/Movement
    stages = human_data.get('stage_expert', np.array([]))
    
    # Filter out label 5 (often used by experts for movement/unscored)
    valid_mask = (stages < 9.0)
    valid_stages = stages[valid_mask]
    
    if len(valid_stages) > 0:
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        efficiency = np.mean(valid_stages > 0)
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = 0.0

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Fragmentation & Stability (Replacing Probabilities) ---
    # These metrics quantify how "broken" the sleep is, which is a key marker.
    if len(valid_stages) > 1:
        # Number of stage transitions
        transitions = np.count_nonzero(np.diff(valid_stages)) / total_hours
        # Wake After Sleep Onset (WASO) proxy: non-zero stages followed by zero
        waso_minutes = (np.count_nonzero(valid_stages == 0) * 30) / 60.0
        # REM Latency (epochs until first REM)
        rem_indices = np.where(valid_stages == 4)[0]
        rem_latency = rem_indices[0] if len(rem_indices) > 0 else 0.0
    else:
        transitions = waso_minutes = rem_latency = 0.0

    features.extend([transitions, waso_minutes, rem_latency])

    return np.array(features)


# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)