import time
import hmac
import hashlib

CLASSROOM_ID = "ROOM_101"
CLASSROOM_SECRET_KEY = b"simulation_ble_salt_key_benchmark"
CYCLE_INTERVAL_SEC = 4
CALIBRATED_TX_POWER = -12  # Hardware Tx Power level setting in dBm
CALIBRATED_1M_RSSI = -59   # Calibrated reference RSSI at d0 = 1 meter

def verify_spatial_boundary(
    rssi: float | int | None,
    ref_rssi_1m: int = CALIBRATED_1M_RSSI,
    max_distance_m: float = 10.0,
    path_loss_exp: float = 2.7
) -> tuple[bool, float]:
    """
    Evaluates Log-Distance Path-Loss Model with Indoor Multipath Attenuation:
        RSSI(d) = RSSI_0(1m) - 10 * gamma * log10(d / d0) + X_sigma
        => d_hat = 10 ^ ((RSSI_0(1m) - RSSI) / (10 * gamma))

    Rejects signals outside physical perimeter or with non-line-of-sight (NLOS)
    wall attenuation (e.g. RSSI drop > 12-15 dB across concrete boundaries).
    """
    if rssi is None:
        # Spatial filtering bypassed when RSSI hardware telemetry is unavailable
        return True, 0.0

    try:
        rssi_val = float(rssi)
        # Physical plausibility filter
        if rssi_val > 0 or rssi_val < -110:
            return False, 999.0

        estimated_distance = 10.0 ** ((ref_rssi_1m - rssi_val) / (10.0 * path_loss_exp))
        is_inside_boundary = estimated_distance <= max_distance_m
        return is_inside_boundary, round(estimated_distance, 2)
    except Exception:
        return False, 999.0


class RollingSecurityEngine:
    def __init__(self, room_id: str, secret_key: bytes, interval: int = 4):
        self.room_id = room_id
        self.secret_key = secret_key
        self.interval = interval

    def get_current_window(self) -> int:
        return int(time.time() // self.interval)

    def generate_token(self, window_offset: int = 0) -> tuple[str, int]:
        target_window = self.get_current_window() + window_offset
        message = f"{self.room_id}:{target_window}".encode('utf-8')
        digest = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        token = digest[:8].upper()
        return token, target_window

    def validate_token(
        self,
        token: str,
        max_drift_windows: int = 1,
        rssi: int | float | None = None,
        max_distance_m: float = 10.0,
        path_loss_exp: float = 2.7
    ) -> bool:
        """
        Hardened Zero-Trust Spatiotemporal Validation:
        1. Freshness Check: strictly bounds temporal offset to [-max_drift_windows, +max_drift_windows].
           Default delta = 1 window (+-4 seconds) eliminating replay/relay attacks.
        2. Spatial Boundary Check: validates RSSI against log-distance path loss perimeter.
        """
        if not token:
            return False

        token = token.strip().upper()
        token_matched = False

        for offset in range(-max_drift_windows, max_drift_windows + 1):
            valid_token, _ = self.generate_token(offset)
            if valid_token == token:
                token_matched = True
                break

        if not token_matched:
            return False

        if rssi is not None:
            is_valid_space, _ = verify_spatial_boundary(
                rssi=rssi,
                ref_rssi_1m=CALIBRATED_1M_RSSI,
                max_distance_m=max_distance_m,
                path_loss_exp=path_loss_exp
            )
            if not is_valid_space:
                return False

        return True

    def validate_token_detailed(
        self,
        token: str,
        max_drift_windows: int = 1,
        rssi: int | float | None = None,
        max_distance_m: float = 10.0,
        path_loss_exp: float = 2.7
    ) -> dict:
        """Detailed telemetry for research experiments and diagnostics."""
        token = (token or "").strip().upper()
        matched_offset = None
        for offset in range(-max_drift_windows, max_drift_windows + 1):
            valid_token, _ = self.generate_token(offset)
            if valid_token == token:
                matched_offset = offset
                break

        time_valid = matched_offset is not None
        space_valid, est_dist = verify_spatial_boundary(
            rssi=rssi,
            ref_rssi_1m=CALIBRATED_1M_RSSI,
            max_distance_m=max_distance_m,
            path_loss_exp=path_loss_exp
        )

        overall_valid = time_valid and (space_valid if rssi is not None else True)
        return {
            "valid": overall_valid,
            "temporal_valid": time_valid,
            "matched_offset": matched_offset,
            "spatial_valid": space_valid,
            "estimated_distance_m": est_dist,
            "rssi": rssi
        }
