from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from neural_decoder.models import build_model


APP_DIR = Path(__file__).parent
ROOT = APP_DIR.parent.parent
ARTIFACTS = ROOT / "artifacts"

NUM_CHANNELS_DEFAULT = int(os.getenv("NUM_CHANNELS", "32"))
WINDOW_SIZE_DEFAULT = int(os.getenv("WINDOW_SIZE", "64"))

app = FastAPI(title="Neural Signal Decoder (Research Demo)")


class InferRequest(BaseModel):
    signal: List[List[float]] = Field(..., description="[num_channels][window_size] array of neural samples")


class InferResponse(BaseModel):
    dx: float
    dy: float
    latency_ms: float


class ModelBundle:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(1)  # stabilize latency on CPU
        self.num_channels = NUM_CHANNELS_DEFAULT
        self.window_size = WINDOW_SIZE_DEFAULT

        # Load normalization stats
        mean = np.zeros(self.num_channels, dtype=np.float32)
        std = np.ones(self.num_channels, dtype=np.float32)
        norm_path = ARTIFACTS / "normalizer.npz"
        if norm_path.exists():
            data = np.load(norm_path)
            mean = data.get("mean", mean)
            std = data.get("std", std)
        self.mean = torch.from_numpy(mean).to(self.device).view(1, -1, 1)
        self.std = torch.from_numpy(std).to(self.device).view(1, -1, 1).clamp_min(1e-6)

        # Load model (prefer scripted)
        scripted_path = ARTIFACTS / "model_scripted.pt"
        if scripted_path.exists():
            self.model = torch.jit.load(str(scripted_path), map_location=self.device)
            # attach for warmup values
            setattr(self.model, "num_channels", self.num_channels)
            setattr(self.model, "window_size", self.window_size)
        else:
            self.model = build_model(self.num_channels, self.window_size).to(self.device)
            pt_path = ARTIFACTS / "model.pt"
            if pt_path.exists():
                state = torch.load(str(pt_path), map_location=self.device)
                self.model.load_state_dict(state["state_dict"])  # type: ignore[index]
        self.model.eval()
        with torch.inference_mode():
            dummy = torch.zeros(1, self.num_channels, self.window_size, device=self.device)
            _ = self._forward(dummy)

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, torch.jit.ScriptModule) or isinstance(self.model, torch.jit.RecursiveScriptModule):
            return self.model(x)  # type: ignore[no-any-return]
        return self.model(x)  # type: ignore[no-any-return]

    @torch.inference_mode()
    def infer(self, signal: np.ndarray) -> np.ndarray:
        # signal: [C, W]
        x = torch.from_numpy(signal).to(self.device).float().unsqueeze(0)  # [1, C, W]
        x = (x - self.mean) / self.std
        out = self._forward(x)  # [1, 2]
        return out.squeeze(0).detach().cpu().numpy()


bundle = ModelBundle()


@app.get("/")
async def root():
    index_html = (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=index_html, status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(bundle.device)}


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    start = time.perf_counter()
    sig = np.asarray(req.signal, dtype=np.float32)
    assert sig.shape == (bundle.num_channels, bundle.window_size), f"expected {(bundle.num_channels, bundle.window_size)}, got {sig.shape}"
    dxdy = bundle.infer(sig)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {"dx": float(dxdy[0]), "dy": float(dxdy[1]), "latency_ms": float(latency_ms)}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            sig = np.asarray(msg.get("signal"), dtype=np.float32)
            if sig.shape != (bundle.num_channels, bundle.window_size):
                await ws.send_json({"error": f"signal shape must be {(bundle.num_channels, bundle.window_size)}"})
                continue
            start = time.perf_counter()
            dxdy = bundle.infer(sig)
            latency_ms = (time.perf_counter() - start) * 1000.0
            await ws.send_json({"dx": float(dxdy[0]), "dy": float(dxdy[1]), "latency_ms": float(latency_ms)})
    except WebSocketDisconnect:
        return
