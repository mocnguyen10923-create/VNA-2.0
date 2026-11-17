"""Ket noi mo hinh AI: Ollama (tren may), Claude API, hoac may chu tuong thich OpenAI."""
import json
import urllib.error
import urllib.request
from urllib.parse import urlparse


class LLMError(Exception):
    pass


def _opener_for(url: str):
    host = (urlparse(url).hostname or "").lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        # Khong di qua proxy he thong khi goi may chu tren chinh laptop
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        return True
    # may chu trong mang noi bo (LAN) cua co quan cung duoc coi la noi bo
    return host.startswith(("10.", "192.168.")) or any(host.startswith(f"172.{i}.") for i in range(16, 32))


OFFLINE_MSG = ("Đang bật chế độ 'Hoàn toàn ngoại tuyến' nên VNA không kết nối Internet. "
               "Hãy dùng mô hình trên máy (Ollama – miễn phí), hoặc tắt chế độ này trong Cài đặt nếu thực sự cần.")


def guard(cfg: dict, url: str):
    if cfg.get("offline_only", True) and not _is_local(url):
        raise LLMError(OFFLINE_MSG)


def _request(url, payload=None, headers=None, timeout=900, method=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method or ("POST" if data else "GET"))
    try:
        with _opener_for(url).open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            msg = json.loads(body).get("error")
            if isinstance(msg, dict):
                msg = msg.get("message")
        except Exception:
            msg = body[:400]
        raise LLMError(f"Máy chủ AI báo lỗi {e.code}: {msg}")
    except urllib.error.URLError as e:
        raise LLMError(f"Không kết nối được máy chủ AI ({url}): {e.reason}")
    except TimeoutError:
        raise LLMError("Máy chủ AI phản hồi quá lâu (hết thời gian chờ).")


def chat(cfg: dict, system: str, messages: list, json_mode: bool = False) -> str:
    """messages: [{'role': 'user'|'assistant', 'content': str}]"""
    backend = cfg.get("backend", "ollama")
    temp = float(cfg.get("temperature", 0.3))
    max_out = int(cfg.get("max_output_tokens", 8000))

    if backend == "claude":
        guard(cfg, "https://api.anthropic.com")
        key = cfg.get("claude_api_key", "").strip()
        if not key:
            raise LLMError("Chưa nhập Claude API key (vào Cài đặt).")
        res = _request(
            "https://api.anthropic.com/v1/messages",
            {
                "model": cfg.get("claude_model") or "claude-sonnet-5-5",
                "max_tokens": max_out,
                "temperature": temp,
                "system": system,
                "messages": messages,
            },
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        return "".join(b.get("text", "") for b in res.get("content", []) if b.get("type") == "text").strip()

    if backend == "openai":
        base = cfg.get("openai_base_url", "").rstrip("/")
        guard(cfg, base)
        headers = {}
        if cfg.get("openai_api_key"):
            headers["Authorization"] = "Bearer " + cfg["openai_api_key"]
        res = _request(
            base + "/chat/completions",
            {
                "model": cfg.get("openai_model") or "local-model",
                "temperature": temp,
                "max_tokens": max_out,
                "messages": [{"role": "system", "content": system}] + messages,
            },
            headers,
        )
        return res["choices"][0]["message"]["content"].strip()

    # Mac dinh: Ollama tren laptop
    base = cfg.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
    guard(cfg, base)
    res = _request(
        base + "/api/chat",
        {
            "model": cfg.get("ollama_model") or "qwen2.5:7b",
            "stream": False,
            "messages": [{"role": "system", "content": system}] + messages,
            **({"format": "json"} if json_mode else {}),
            "options": {"temperature": temp, "num_ctx": int(cfg.get("ollama_num_ctx", 16384)), "num_predict": max_out},
        },
    )
    text = res.get("message", {}).get("content", "")
    # Mot so mo hinh suy luan tra ve <think>...</think>: bo di
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def status(cfg: dict) -> dict:
    backend = cfg.get("backend", "ollama")
    out = {"backend": backend, "ok": False, "message": "", "models": []}
    if backend == "ollama":
        base = cfg.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
        try:
            res = _request(base + "/api/tags", timeout=4)
            names = [m.get("name", "") for m in res.get("models", [])]
            out["models"] = names
            want = cfg.get("ollama_model", "")
            if any(n == want or n.split(":")[0] == want for n in names):
                out["ok"] = True
                out["message"] = f"Đang dùng mô hình trên máy: {want} · miễn phí" + (" · ngoại tuyến" if cfg.get("offline_only", True) else "")
            else:
                out["message"] = (f"Ollama đang chạy nhưng chưa có mô hình '{want}'. "
                                  f"Mở Command Prompt và chạy: ollama pull {want}")
        except LLMError:
            out["message"] = ("Chưa thấy Ollama. Hãy cài Ollama (https://ollama.com) rồi chạy: "
                              f"ollama pull {cfg.get('ollama_model')} — hoặc chuyển sang Claude API trong Cài đặt.")
    elif backend == "claude" and cfg.get("offline_only", True):
        out["message"] = "Chế độ ngoại tuyến đang bật: không dùng được Claude API. Chọn Ollama (miễn phí) hoặc tắt chế độ ngoại tuyến."
    elif backend == "claude":
        out["ok"] = bool(cfg.get("claude_api_key"))
        out["message"] = (f"Đang dùng Claude API ({cfg.get('claude_model')})" if out["ok"]
                          else "Chưa nhập Claude API key (vào Cài đặt).")
    else:
        out["ok"] = bool(cfg.get("openai_base_url"))
        out["message"] = f"Đang dùng máy chủ: {cfg.get('openai_base_url')}"
    return out
