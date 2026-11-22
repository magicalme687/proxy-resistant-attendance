"""
Script: run_celeba_spoof_classifier_benchmark.py
Extracts 16-D Multi-Domain Spectral-Chrominance Forensic Feature Vectors
from CelebA-Spoof physical test crops (Partition 0).
Compares:
  - Baseline A: Fixed Uncalibrated Heuristics (ACER ~ 70.61%, AUC ~ 0.4962)
  - Proposed Stage 1: Calibrated Multi-Spectral Forensic Classifier (5-Fold Stratified CV)
Logs: APCER, BPCER, ACER, EER, ROC-AUC, 95% bootstrap CIs, and inference latency.
"""
import os, sys, time, json, io
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pyarrow.parquet as pq
from scipy.stats import kurtosis, entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET_FILE = os.path.join(ROOT_DIR, "datasets", "celeba_spoof", "test-00000-of-00010.parquet")
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_16d_features(cv_img):
    """
    Extracts a 16-dimensional Multi-Domain Spectral-Chrominance Forensic Feature Vector:
      1-3: Multi-scale Laplacian variance (original, 1/2, 1/4 scale)
      4-5: Sobel gradient mean energy and kurtosis
      6-9: 2D-FFT azimuthal radial ring energy ratios (4 frequency octaves)
      10-14: HSV & YCbCr chrominance moments (Sat mean/std, Cb std, Cr std, Hue entropy)
      15-16: Specular highlight ratio and top-5% luminance variance
    """
    h, w = cv_img.shape[:2]
    # Resize to standard canonical size for fast reproducible feature extraction
    if h != 224 or w != 224:
        cv_img = cv2.resize(cv_img, (224, 224), interpolation=cv2.INTER_AREA)
    h, w = 224, 224
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # 1-3: Multi-scale Laplacian variance
    lap1 = cv2.Laplacian(gray, cv2.CV_64F)
    var1 = float(np.var(lap1))
    
    gray_half = cv2.pyrDown(gray)
    lap2 = cv2.Laplacian(gray_half, cv2.CV_64F)
    var2 = float(np.var(lap2))
    
    gray_quarter = cv2.pyrDown(gray_half)
    lap3 = cv2.Laplacian(gray_quarter, cv2.CV_64F)
    var3 = float(np.var(lap3))
    
    f1_lap_orig = np.log1p(var1)
    f2_lap_half = np.log1p(var2)
    f3_lap_quart = np.log1p(var3)
    
    # 4-5: Sobel gradient energy & kurtosis
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.hypot(sobel_x, sobel_y)
    f4_sobel_mean = float(np.mean(sobel_mag))
    f5_sobel_kurt = float(kurtosis(sobel_mag.ravel()))
    
    # 6-9: 2D-FFT Azimuthal Radial Ring Energy Ratios
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f_shift)
    total_mag = float(np.sum(mag)) + 1e-7
    
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.hypot(x - cx, y - cy)
    max_r = np.hypot(cx, cy)
    r_norm = dist_from_center / max_r
    
    # 4 concentric radial frequency bands
    ring1 = (r_norm < 0.15)
    ring2 = (r_norm >= 0.15) & (r_norm < 0.35)
    ring3 = (r_norm >= 0.35) & (r_norm < 0.65)
    ring4 = (r_norm >= 0.65)
    
    f6_fft_band1 = float(np.sum(mag[ring1])) / total_mag
    f7_fft_band2 = float(np.sum(mag[ring2])) / total_mag
    f8_fft_band3 = float(np.sum(mag[ring3])) / total_mag
    f9_fft_band4 = float(np.sum(mag[ring4])) / total_mag
    
    # 10-14: HSV & YCbCr Chrominance Moments
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    f10_sat_mean = float(np.mean(s_chan))
    f11_sat_std = float(np.std(s_chan))
    
    ycbcr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YCrCb)
    y_chan, cr_chan, cb_chan = cv2.split(ycbcr)
    f12_cb_std = float(np.std(cb_chan))
    f13_cr_std = float(np.std(cr_chan))
    
    # Hue histogram entropy (richness of color gamut)
    h_hist, _ = np.histogram(h_chan, bins=16, range=(0, 180), density=True)
    f14_hue_entropy = float(entropy(h_hist + 1e-7))
    
    # 15-16: Specular highlights & High-luminance variance
    # Glass and phone screen replays concentrate intense specular reflections
    glare_mask = (v_chan > 240) & (s_chan < 35)
    f15_glare_ratio = float(np.sum(glare_mask)) / float(h * w)
    
    top_5_pct_thresh = np.percentile(gray, 95)
    top_pixels = gray[gray >= top_5_pct_thresh]
    f16_top_lum_var = float(np.var(top_pixels)) if len(top_pixels) > 1 else 0.0
    
    return [
        f1_lap_orig, f2_lap_half, f3_lap_quart,
        f4_sobel_mean, f5_sobel_kurt,
        f6_fft_band1, f7_fft_band2, f8_fft_band3, f9_fft_band4,
        f10_sat_mean, f11_sat_std,
        f12_cb_std, f13_cr_std, f14_hue_entropy,
        f15_glare_ratio, f16_top_lum_var
    ]

def evaluate_uncalibrated_heuristics(cv_img):
    """
    Evaluates the uncalibrated heuristic rule from the baseline prototype.
    """
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude_spectrum = np.abs(f_shift)
    cy, cx = h // 2, w // 2
    r_inner = min(h, w) // 6
    y, x = np.ogrid[:h, :w]
    mask_inner = (x - cx)**2 + (y - cy)**2 <= r_inner**2
    total_energy = float(np.sum(magnitude_spectrum)) + 1e-7
    hf_ratio = float(np.sum(magnitude_spectrum[~mask_inner])) / total_energy
    
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    _, s_chan, v_chan = cv2.split(hsv)
    glare_ratio = float(np.sum((v_chan > 245) & (s_chan < 30))) / float(h * w)
    
    import math
    sharpness_score = 1.0 / (1.0 + math.exp(-0.05 * (lap_var - 45.0)))
    if hf_ratio > 0.45:
        fft_score = max(0.0, 1.0 - (hf_ratio - 0.45) * 4.0)
    else:
        fft_score = min(1.0, hf_ratio / (0.45 * 0.75))
    glare_penalty = max(0.0, 1.0 - (glare_ratio / 0.08))
    color_naturalness = min(1.0, float(np.std(s_chan)) / 35.0)
    
    liveness_score = (
        0.35 * sharpness_score +
        0.30 * fft_score +
        0.20 * glare_penalty +
        0.15 * color_naturalness
    )
    is_live = (
        (liveness_score >= 0.55) and
        (14.0 <= lap_var <= 450.0) and
        (hf_ratio <= 0.45) and
        (glare_ratio <= 0.08)
    )
    return is_live, liveness_score

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])

def bootstrap_ci(y_true, y_scores, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    eers = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[boot_idx])) < 2:
            continue
        b_fpr, b_tpr, _ = roc_curve(y_true[boot_idx], y_scores[boot_idx])
        b_auc = auc(b_fpr, b_tpr)
        b_fnr = 1.0 - b_tpr
        b_eer_idx = np.nanargmin(np.abs(b_fpr - b_fnr))
        b_eer = (b_fpr[b_eer_idx] + b_fnr[b_eer_idx]) / 2.0
        aucs.append(b_auc)
        eers.append(b_eer)
    auc_ci = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    eer_ci = [float(np.percentile(eers, 2.5)), float(np.percentile(eers, 97.5))]
    return auc_ci, eer_ci

def main():
    print("=" * 75)
    print("CELEBA-SPOOF 16-D SPECTRAL-CHROMINANCE FORENSIC CLASSIFIER BENCHMARK")
    print("=" * 75)
    
    cache_path = os.path.join(RESULTS_DIR, "celeba_features_cache.npz")
    if os.path.exists(cache_path):
        print(f"Loading precomputed features and labels from cache: {cache_path}")
        cache_data = np.load(cache_path)
        X = cache_data["X"]
        y = cache_data["y"]
        heuristic_preds = list(cache_data["heuristic_preds"])
        heuristic_scores = list(cache_data["heuristic_scores"])
        feature_times = list(cache_data["feature_times"])
    else:
        table = pq.read_table(PARQUET_FILE)
        df = table.select(['cropped_image', 'labels', 'labelNames']).to_pandas()
        print(f"Loaded {len(df)} samples from CelebA-Spoof Partition 0.")
        
        X_features = []
        y_labels = [] # 1 = Live (bona fide), 0 = Spoof
        heuristic_preds = []
        heuristic_scores = []
        valid_indices = []
        
        t0 = time.time()
        feature_times = []
        
        for idx, row in df.iterrows():
            c_img = row['cropped_image']
            if c_img is None or not isinstance(c_img, dict) or not c_img.get('bytes'):
                continue
            img = Image.open(io.BytesIO(c_img['bytes']))
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            t_start = time.perf_counter()
            feats = extract_16d_features(cv_img)
            t_end = time.perf_counter()
            feature_times.append((t_end - t_start) * 1000.0) # in ms
            
            h_live, h_score = evaluate_uncalibrated_heuristics(cv_img)
            
            # Ground truth: labels: 0 = live, 1 = spoof (in CelebA-Spoof parquet)
            # We define positive class = 1 (Live/Bona fide), 0 = Spoof (Attack)
            is_live_gt = 1 if row['labels'] == 0 else 0
            
            X_features.append(feats)
            y_labels.append(is_live_gt)
            heuristic_preds.append(1 if h_live else 0)
            heuristic_scores.append(h_score)
            valid_indices.append(idx)
            
            if len(X_features) % 1500 == 0:
                print(f"Extracted features for {len(X_features)} images ({time.time() - t0:.1f}s)...")
                
        X = np.array(X_features, dtype=np.float32)
        y = np.array(y_labels, dtype=np.int32)
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            heuristic_preds=np.array(heuristic_preds),
            heuristic_scores=np.array(heuristic_scores),
            feature_times=np.array(feature_times)
        )
        print(f"Saved extracted features cache to: {cache_path}")
    
    n_samples = len(y)
    n_live = int(np.sum(y == 1))
    n_spoof = int(np.sum(y == 0))
    print(f"\nDataset Extracted: {n_samples} valid crops | Live (Bona Fide): {n_live} | Spoofs: {n_spoof}")
    print(f"Feature Extraction Latency: Median = {np.median(feature_times):.3f} ms, Mean = {np.mean(feature_times):.3f} ms")
    
    # -------------------------------------------------------------
    # BASELINE A: Fixed Uncalibrated Heuristic Rules
    # -------------------------------------------------------------
    h_preds = np.array(heuristic_preds)
    h_scores = np.array(heuristic_scores)
    
    # APCER: Spoofs classified as Live
    # BPCER: Live classified as Spoof
    spoof_mask = (y == 0)
    live_mask = (y == 1)
    
    base_apcer = float(np.sum(h_preds[spoof_mask] == 1) / n_spoof) * 100.0
    base_bpcer = float(np.sum(h_preds[live_mask] == 0) / n_live) * 100.0
    base_acer = (base_apcer + base_bpcer) / 2.0
    base_fpr, base_tpr, _ = roc_curve(y, h_scores)
    base_auc = float(auc(base_fpr, base_tpr))
    base_eer, base_eer_thresh = compute_eer(y, h_scores)
    
    print("\n" + "=" * 50)
    print("BASELINE A: FIXED UNCALIBRATED HEURISTIC RULES")
    print("=" * 50)
    print(f"  APCER:     {base_apcer:.2f}% ({np.sum(h_preds[spoof_mask] == 1)}/{n_spoof})")
    print(f"  BPCER:     {base_bpcer:.2f}% ({np.sum(h_preds[live_mask] == 0)}/{n_live})")
    print(f"  ACER:      {base_acer:.2f}%")
    print(f"  ROC-AUC:   {base_auc:.4f}")
    print(f"  EER:       {base_eer * 100.0:.2f}% at threshold {base_eer_thresh:.4f}")
    
    # -------------------------------------------------------------
    # PROPOSED STAGE 1: Calibrated Multi-Spectral Forensic Classifier
    # 5-Fold Stratified Cross-Validation
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PROPOSED STAGE 1: CALIBRATED 16-D MULTI-SPECTRAL CLASSIFIER")
    print("=" * 50)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(n_samples, dtype=np.float32)
    clf_times = []
    
    # We evaluate HistGradientBoostingClassifier (ultra-fast, robust, sub-millisecond on CPU)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        base = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42 + fold
        )
        # Platt (sigmoid) calibration fitted with an internal 3-fold split of the
        # training fold only -- the validation fold is never seen during calibration.
        clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        clf.fit(X[train_idx], y[train_idx])
        
        t_c0 = time.perf_counter()
        val_probs = clf.predict_proba(X[val_idx])[:, 1]
        t_c1 = time.perf_counter()
        clf_times.append((t_c1 - t_c0) / len(val_idx) * 1000.0) # per-sample ms
        
        oof_probs[val_idx] = val_probs
        
    prop_fpr, prop_tpr, prop_thresholds = roc_curve(y, oof_probs)
    prop_auc = float(auc(prop_fpr, prop_tpr))
    prop_eer, prop_eer_thresh = compute_eer(y, oof_probs)
    
    # Evaluate at EER threshold
    prop_preds_eer = (oof_probs >= prop_eer_thresh).astype(int)
    prop_apcer = float(np.sum(prop_preds_eer[spoof_mask] == 1) / n_spoof) * 100.0
    prop_bpcer = float(np.sum(prop_preds_eer[live_mask] == 0) / n_live) * 100.0
    prop_acer = (prop_apcer + prop_bpcer) / 2.0
    
    # Also evaluate at strict high-security threshold (APCER <= 5.0%)
    target_apcer_idx = np.where(prop_fpr <= 0.05)[0]
    strict_thresh = float(prop_thresholds[target_apcer_idx[-1]]) if len(target_apcer_idx) > 0 else 0.5
    prop_preds_strict = (oof_probs >= strict_thresh).astype(int)
    strict_apcer = float(np.sum(prop_preds_strict[spoof_mask] == 1) / n_spoof) * 100.0
    strict_bpcer = float(np.sum(prop_preds_strict[live_mask] == 0) / n_live) * 100.0
    strict_acer = (strict_apcer + strict_bpcer) / 2.0
    
    print("Computing 1,000 bootstrap confidence intervals...")
    auc_ci, eer_ci = bootstrap_ci(y, oof_probs, n_bootstraps=1000, seed=42)
    
    total_cpu_latency_ms = float(np.median(feature_times)) + float(np.mean(clf_times))
    
    print(f"  ROC-AUC:   {prop_auc:.4f} (95% CI [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}])")
    print(f"  EER:       {prop_eer * 100.0:.2f}% (95% CI [{eer_ci[0] * 100.0:.2f}%, {eer_ci[1] * 100.0:.2f}%]) at threshold {prop_eer_thresh:.4f}")
    print(f"  At EER Operating Point:")
    print(f"    APCER:   {prop_apcer:.2f}% ({np.sum(prop_preds_eer[spoof_mask] == 1)}/{n_spoof})")
    print(f"    BPCER:   {prop_bpcer:.2f}% ({np.sum(prop_preds_eer[live_mask] == 0)}/{n_live})")
    print(f"    ACER:    {prop_acer:.2f}%")
    print(f"  At High-Security Operating Point (APCER <= 5.0%):")
    print(f"    Threshold: {strict_thresh:.4f}")
    print(f"    APCER:   {strict_apcer:.2f}%")
    print(f"    BPCER:   {strict_bpcer:.2f}%")
    print(f"    ACER:    {strict_acer:.2f}%")
    print(f"  Total CPU Latency (Feature Extr + Classification): {total_cpu_latency_ms:.3f} ms")
    
    summary = {
        "dataset": "CelebA-Spoof Test Partition 0",
        "sample_count": n_samples,
        "live_count": n_live,
        "spoof_count": n_spoof,
        "baseline_uncalibrated_heuristics": {
            "apcer_pct": round(base_apcer, 2),
            "bpcer_pct": round(base_bpcer, 2),
            "acer_pct": round(base_acer, 2),
            "roc_auc": round(base_auc, 4),
            "eer_pct": round(base_eer * 100.0, 2),
            "eer_threshold": round(base_eer_thresh, 4)
        },
        "proposed_calibrated_multispectral_classifier": {
            "features_dim": 16,
            "validation_scheme": "5-Fold Stratified Cross-Validation",
            "classifier": "CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=100), method=sigmoid [Platt], cv=3) fitted inside each outer fold",
            "roc_auc": round(prop_auc, 4),
            "roc_auc_ci_95": [round(auc_ci[0], 4), round(auc_ci[1], 4)],
            "eer_pct": round(prop_eer * 100.0, 2),
            "eer_ci_95": [round(eer_ci[0] * 100.0, 2), round(eer_ci[1] * 100.0, 2)],
            "eer_threshold": round(prop_eer_thresh, 4),
            "eer_operating_point": {
                "apcer_pct": round(prop_apcer, 2),
                "bpcer_pct": round(prop_bpcer, 2),
                "acer_pct": round(prop_acer, 2)
            },
            "high_security_operating_point": {
                "threshold": round(strict_thresh, 4),
                "apcer_pct": round(strict_apcer, 2),
                "bpcer_pct": round(strict_bpcer, 2),
                "acer_pct": round(strict_acer, 2)
            },
            "latency": {
                "feature_extraction_ms_median": round(float(np.median(feature_times)), 3),
                "classifier_inference_ms_mean": round(float(np.mean(clf_times)), 3),
                "total_cpu_latency_ms": round(total_cpu_latency_ms, 3)
            }
        }
    }
    
    out_json = os.path.join(RESULTS_DIR, "real_celeba_spoof_classifier_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {out_json}")

    # Export per-sample out-of-fold scores for independent reviewer audit
    scores_df = pd.DataFrame({
        "sample_id": np.arange(n_samples),
        "ground_truth": y,
        "predicted_live_prob": np.round(oof_probs, 5),
        "heuristic_pred": h_preds,
        "heuristic_score": np.round(h_scores, 5)
    })
    out_csv = os.path.join(RESULTS_DIR, "real_celeba_spoof_scores.csv")
    scores_df.to_csv(out_csv, index=False)
    print(f"Saved per-sample out-of-fold scores to: {out_csv}")

if __name__ == "__main__":
    main()
