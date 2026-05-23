"""
—  Real-time Seismograph WebSocket Server
=========================================================

Install tambahan (opsional, untuk performa):
  pip install quakemigrate
"""

import asyncio, json, threading, time, collections
import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import zoom
import websockets, os
from datetime import datetime, timezone

# 
try:
    from quakemigrate.core.lib import recursive_sta_lta as _qm_stalta
    _QM_AVAILABLE = True
    print("[ONSET] QuakeMigrate C library loaded → recursive_sta_lta aktif")
except ImportError:
    from obspy.signal.trigger import classic_sta_lta as _obs_stalta
    _QM_AVAILABLE = False
    print("[ONSET] QuakeMigrate tidak ditemukan → fallback ke obspy classic_sta_lta")

from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.signal.trigger import trigger_onset
from obspy.signal.filter import bandpass as obspy_bandpass



STATIONS = [
 
    {"net":"GE","sta":"UGM",  "cha":"SHZ","label":"WanaGAMA",
     "thr_on":4.5,"thr_off":0.6,
     "bandpass":(1.5,8.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"JAGI", "cha":"BHZ","label":"Banyuwangi",
     "thr_on":4.8,"thr_off":0.5,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"BBJI", "cha":"BHZ","label":"Garut",
     "thr_on":3.2,"thr_off":0.5,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"SMRI", "cha":"BHZ","label":"Semarang",
     "thr_on":7.5,"thr_off":0.9,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"PLAI", "cha":"BHZ","label":"Sumbawa",
     "thr_on":4.0,"thr_off":0.6,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"MMRI", "cha":"BHZ","label":"Maumere",
     "thr_on":3.8,"thr_off":0.6,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"SOEI", "cha":"BHZ","label":"Soe Timor",
     "thr_on":3.5,"thr_off":0.6,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"SAUI", "cha":"SHZ","label":"Tanibar",
     "thr_on":7.5,"thr_off":0.6,
     "bandpass":(1.0,8.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"BNDI", "cha":"BHZ","label":"BandaNeira",
     "thr_on":6.5,"thr_off":0.6,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"FAKI", "cha":"BHZ","label":"FakFak",
     "thr_on":6.5,"thr_off":0.6,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"SANI", "cha":"LHZ","label":"Maluku",
     "thr_on":7.5,"thr_off":0.6,
     "bandpass":(0.5,5.0),"sta_sec":1.0,"lta_sec":20.0}, 
    {"net":"GE","sta":"TNTI", "cha":"BHZ","label":"Ternate",
     "thr_on":7.0,"thr_off":0.7,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"LUWI", "cha":"BHZ","label":"Luwu Sulawesi",
     "thr_on":7.8,"thr_off":0.6,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"TOLI2","cha":"BHZ","label":"ToliToli",
     "thr_on":7.0,"thr_off":0.7,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"LHMI", "cha":"BHZ","label":"Aceh",
     "thr_on":8.0,"thr_off":0.7,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"GSI",  "cha":"BHZ","label":"Nias",
     "thr_on":7.8,"thr_off":0.7,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"GE","sta":"MNAI", "cha":"BHZ","label":"Bengkulu",
     "thr_on":7.8,"thr_off":0.6,
     "bandpass":(1.0,10.0),"sta_sec":0.5,"lta_sec":12.0},
    {"net":"AU","sta":"XMI",  "cha":"BHZ","label":"Christmas Island",
     "thr_on":4.0,"thr_off":0.6,
     "bandpass":(0.8,8.0),"sta_sec":0.5,"lta_sec":12.0,
     "server":"iris"},
]

#Konstanta spectrogram
WINDOW_SEC = 120    
PUSH_SEC   = 1     
N_FREQ     = 24     
N_TIME     = 150    


ONSET_PUSH_FS = 50 

#Server SeedLink
GEOFON_HOST = "geofon.gfz-potsdam.de"
GEOFON_PORT = 18000
IRIS_HOST   = "rtserve.iris.washington.edu"
IRIS_PORT   = 18000

STATIONS_GEOFON = [s for s in STATIONS if s.get("server", "geofon") == "geofon"]
STATIONS_IRIS   = [s for s in STATIONS if s.get("server") == "iris"]

# Buffer per stasiun 
buffers = {
    s["sta"]: {
        "data":       collections.deque(maxlen=WINDOW_SEC * 100),
        "sr":         100.0,
        "triggered":  False,
        "magnitude":  None,
        "status":     "init",
      
        "onset_ds":   [],          
        "onset_thr":  s["thr_on"], 
        "p_pick_ms":  None,        
    } for s in STATIONS
}

lock = threading.Lock()

#Fungsi komputasi onset
def compute_onset(arr: np.ndarray, sr: float, cfg: dict):
    """
    Hitung onset function STA/LTA.
    """
    # Filter bandpass per stasiun
    flo, fhi = cfg.get("bandpass", (1.0, 10.0))
    nyq = sr / 2.0
    fhi = min(fhi, nyq * 0.95)  

    try:
        filtered = obspy_bandpass(arr.astype(np.float64), flo, fhi, sr,
                                  corners=2, zerophase=True)
    except Exception:
        filtered = arr.astype(np.float64)

    nsta = max(1, int(cfg.get("sta_sec", 0.5) * sr))
    nlta = max(1, int(cfg.get("lta_sec", 12.0) * sr))

    if _QM_AVAILABLE:
      
        onset = _qm_stalta(filtered, nsta, nlta)
    else:
        # Fallback ObsPy
        onset = _obs_stalta(filtered, nsta, nlta)

    return onset


def downsample_onset(onset: np.ndarray, sr: float, target_fs: int = ONSET_PUSH_FS):
    """Downsample onset ke target_fs agar hemat bandwidth WebSocket."""
    factor = max(1, int(sr / target_fs))
    ds = onset[::factor]
    # Clip 
    ds = np.clip(ds, 0, 50).tolist()
    return ds


def find_p_pick(onset: np.ndarray, sr: float, thr_on: float):
    """
    Cari onset P pertama (offset dalam ms dari awal array).
    Return None jika tidak ada trigger.
    """
    on_off = trigger_onset(onset, thr_on, thr_on * 0.2)
    if len(on_off) == 0:
        return None
    first_on = int(on_off[0][0])
    return int(first_on * 1000 / sr)  


#SeedLink Client 
class MultiStationClient(EasySeedLinkClient):

    def on_data(self, trace):
        sta = trace.stats.station
        if sta not in buffers:
            return

        # stasiun
        cfg = next((s for s in STATIONS if s["sta"] == sta), {})

        with lock:
            buf = buffers[sta]
            sr  = float(trace.stats.sampling_rate)
            buf["sr"] = sr

            # Resize buffer 
            new_maxlen = int(WINDOW_SEC * sr)
            if buf["data"].maxlen != new_maxlen:
                buf["data"] = collections.deque(buf["data"], maxlen=new_maxlen)

            buf["data"].extend(trace.data.tolist())
            buf["status"] = "live"

            arr = np.array(buf["data"])
            min_samples = int(sr * 20)
            if len(arr) < min_samples:
                return

            # Hitung onset
            try:
                onset = compute_onset(arr, sr, cfg)

                # Downsample
                buf["onset_ds"]  = downsample_onset(onset, sr)
                buf["onset_thr"] = cfg.get("thr_on", 5.0)

                # Trigger detection
                on_off = trigger_onset(onset, cfg.get("thr_on", 5.0),
                                              cfg.get("thr_off", 0.8))
                buf["triggered"] = len(on_off) > 0

                # P-pick
                buf["p_pick_ms"] = find_p_pick(onset, sr, cfg.get("thr_on", 5.0))

                # Estimasi magnitude
                if buf["triggered"]:
                    peak = float(np.max(np.abs(arr[-int(sr * 10):])))
                    if peak > 0:
                        import math
                        delta = 8.0 * 500 / 111190  
                        ml = np.log10(peak) + 3 * np.log10(8.0 * delta) - 2.92
                        buf["magnitude"] = round(float(ml), 2)
                else:
                    buf["magnitude"] = None

            except Exception as e:
                print(f"[{sta}] Onset error: {e}")

    def on_seedlink_error(self, error=None):
        print(f"[SeedLink Error] {error}")


# Thread runner SeedLink
def _run_single_seedlink(host, port, station_list, server_name):
    """Jalankan satu client SeedLink dengan auto-reconnect."""
    while True:
        try:
            client = MultiStationClient(f"{host}:{port}")
            for cfg in station_list:
                try:
                    client.select_stream(cfg["net"], cfg["sta"], cfg["cha"])
                    print(f"  [{server_name}] subscribed: {cfg['net']}.{cfg['sta']}.{cfg['cha']}")
                except Exception as e:
                    print(f"  [{server_name}] skip {cfg['sta']}: {e}")
            print(f"[{server_name}] Connected → {host}:{port}")
            client.run()
        except Exception as e:
            print(f"[{server_name}] Reconnect in 10s: {e}")
            time.sleep(10)


threading.Thread(
    target=lambda: _run_single_seedlink(GEOFON_HOST, GEOFON_PORT, STATIONS_GEOFON, "GEOFON"),
    daemon=True
).start()

threading.Thread(
    target=lambda: _run_single_seedlink(IRIS_HOST, IRIS_PORT, STATIONS_IRIS, "IRIS"),
    daemon=True
).start()


#  Komputasi spectrogram
def compute_spec(data, sr):
    """
    Hitung spectrogram uint8 (N_FREQ × N_TIME).
    Return list kosong jika data tidak cukup.
    """
    arr = np.array(data)
    if len(arr) < sr * 4:
        return []
    try:
        nperseg  = int(sr * 4)
        noverlap = int(nperseg * 0.75)
        f, _, Sxx = scipy_signal.spectrogram(
            arr, fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann'
        )
        mask = (f >= 0.5) & (f <= 10)
        Sxx  = Sxx[mask]
        if Sxx.size == 0:
            return []
        Sxx  = 10 * np.log10(Sxx + 1e-10)
        noise = np.median(Sxx)
        Sxx  = np.clip((Sxx - noise) / 50.0, 0, 1)
        Sxx  = zoom(Sxx, (N_FREQ / Sxx.shape[0], N_TIME / Sxx.shape[1]), order=1)
        return (Sxx * 255).astype(np.uint8).tolist()
    except Exception as e:
        print(f"Spec error: {e}")
        return []


#WebSocket handler 
async def handler(websocket):
    print(f"[WS] Client connected: {websocket.remote_address}")
    try:
        while True:
            now_ts = datetime.now(timezone.utc).timestamp()

            with lock:
                payload = []
                for cfg in STATIONS:
                    sta = cfg["sta"]
                    buf = buffers[sta]
                    spec = compute_spec(buf["data"], buf["sr"])

                    payload.append({
                        "station":      sta,
                        "label":        cfg["label"],
                        "spec":         spec,
                        "timestamp":    now_ts,
                        "window_sec":   WINDOW_SEC,
                        "triggered":    buf["triggered"],
                        "magnitude":    buf["magnitude"],
                        "status":       buf["status"],
                        "sr":           buf["sr"],
                        
                        "onset_values": buf["onset_ds"],   
                        "onset_thr":    buf["onset_thr"],  
                        "p_pick_ms":    buf["p_pick_ms"],  
                    })

            await websocket.send(json.dumps(payload))
            await asyncio.sleep(PUSH_SEC)

    except websockets.exceptions.ConnectionClosed:
        print(f"[WS] Client disconnected")


# ── Entry point ──────────────────────────────────────────────────────────────
async def main():
    print("[Server] Waiting for SeedLink data (10s)...")
    await asyncio.sleep(10)

    port = int(os.environ.get("PORT", 8765))
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"[Server] WebSocket seismograf running on port {port}")
        await asyncio.Future()

asyncio.run(main())
