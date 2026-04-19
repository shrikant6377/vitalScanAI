"""
utils/iot_simulator.py
======================
Simulates real-time IoT sensor data for heart rate and body temperature.
Mimics a wearable biosensor / body-scanning device with realistic noise,
drift, and physiological variation patterns.
"""

import random
import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SensorReading:
    heart_rate: float           # bpm
    body_temperature: float     # °C
    spo2: float                 # oxygen saturation %
    systolic_bp: float          # mmHg
    diastolic_bp: float         # mmHg
    timestamp: float = field(default_factory=time.time)
    quality_score: float = 1.0  # signal quality 0-1

    def to_dict(self) -> dict:
        return {
            "heart_rate": round(self.heart_rate, 1),
            "body_temperature": round(self.body_temperature, 2),
            "spo2": round(self.spo2, 1),
            "systolic_bp": round(self.systolic_bp, 0),
            "diastolic_bp": round(self.diastolic_bp, 0),
            "timestamp": self.timestamp,
            "quality_score": round(self.quality_score, 2),
        }


class IoTHealthSensor:
    """
    Simulates a body-scanning IoT device.

    Modes:
      - 'healthy'  : normal physiological ranges
      - 'fever'    : elevated temperature & HR
      - 'cardiac'  : arrhythmic / elevated HR patterns
      - 'hypotension': low BP scenario
      - 'random'   : randomly chosen per call
    """

    PROFILES = {
        "healthy": {
            "hr_base": 72, "hr_std": 4,
            "temp_base": 36.8, "temp_std": 0.15,
            "spo2_base": 98.0, "spo2_std": 0.5,
            "sbp_base": 120, "dbp_base": 80,
        },
        "fever": {
            "hr_base": 98, "hr_std": 7,
            "temp_base": 38.9, "temp_std": 0.4,
            "spo2_base": 96.5, "spo2_std": 0.8,
            "sbp_base": 130, "dbp_base": 85,
        },
        "cardiac": {
            "hr_base": 110, "hr_std": 18,
            "temp_base": 37.1, "temp_std": 0.2,
            "spo2_base": 95.0, "spo2_std": 1.2,
            "sbp_base": 145, "dbp_base": 92,
        },
        "hypotension": {
            "hr_base": 95, "hr_std": 6,
            "temp_base": 36.4, "temp_std": 0.2,
            "spo2_base": 97.0, "spo2_std": 0.6,
            "sbp_base": 90, "dbp_base": 58,
        },
    }

    def __init__(self, mode: str = "healthy", seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.mode = mode
        self._tick = 0            # internal clock for drift simulation
        self._drift_hr = 0.0
        self._drift_temp = 0.0

    def _apply_drift(self):
        """Slow sinusoidal drift to mimic physiological cycles."""
        self._tick += 1
        self._drift_hr = 3.0 * math.sin(self._tick * 0.05)
        self._drift_temp = 0.1 * math.sin(self._tick * 0.02)

    def _add_noise(self, value: float, std: float, clip_lo: float, clip_hi: float) -> float:
        noisy = value + random.gauss(0, std)
        return max(clip_lo, min(clip_hi, noisy))

    def read(self) -> SensorReading:
        """Return a single sensor reading snapshot."""
        profile_key = random.choice(list(self.PROFILES)) if self.mode == "random" else self.mode
        p = self.PROFILES.get(profile_key, self.PROFILES["healthy"])

        self._apply_drift()

        hr = self._add_noise(p["hr_base"] + self._drift_hr, p["hr_std"], 35, 210)
        temp = self._add_noise(p["temp_base"] + self._drift_temp, p["temp_std"], 34.5, 42.5)
        spo2 = self._add_noise(p["spo2_base"], p["spo2_std"], 70.0, 100.0)
        sbp = self._add_noise(p["sbp_base"], 6, 60, 220)
        dbp = self._add_noise(p["dbp_base"], 4, 40, 140)

        # Occasionally simulate signal dropout
        quality = 1.0 if random.random() > 0.05 else round(random.uniform(0.5, 0.85), 2)

        return SensorReading(
            heart_rate=hr,
            body_temperature=temp,
            spo2=spo2,
            systolic_bp=sbp,
            diastolic_bp=dbp,
            quality_score=quality,
        )

    def stream(self, n_readings: int = 10, interval_s: float = 0.0):
        """Yield n_readings sequentially (optionally spaced by interval_s)."""
        for _ in range(n_readings):
            yield self.read()
            if interval_s > 0:
                time.sleep(interval_s)

    def batch_readings(self, n: int = 10) -> list[dict]:
        """Return list of n reading dicts (no sleep)."""
        return [r.to_dict() for r in self.stream(n)]


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== IoT Sensor Simulation ===\n")
    for mode in ["healthy", "fever", "cardiac"]:
        sensor = IoTHealthSensor(mode=mode, seed=0)
        r = sensor.read()
        print(f"[{mode.upper():12s}]  HR={r.heart_rate:5.1f} bpm  "
              f"Temp={r.body_temperature:.2f}°C  "
              f"SpO2={r.spo2:.1f}%  "
              f"BP={r.systolic_bp:.0f}/{r.diastolic_bp:.0f} mmHg  "
              f"Quality={r.quality_score:.2f}")
