"""
Streaming text-generation API + a tiny web UI.

    pip install fastapi uvicorn
    uvicorn serve:app --port 8000        # then open http://localhost:8000
"""
import os
import threading
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from inference import Engine

engine = Engine(os.environ.get("CKPT", "out/ckpt_best.pt"),
                os.environ.get("TOKENIZER", "data/tokenizer.json"))
lock = threading.Lock()  # one generation at a time (single model instance)
app = FastAPI(title="Mini GPT", description="From-scratch Llama-style LM trained on TinyStories")


class GenRequest(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = Field(200, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0, le=4096)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)


def _sampling(r: GenRequest):
    return dict(temperature=r.temperature, top_k=r.top_k, top_p=r.top_p,
                repetition_penalty=r.repetition_penalty)


@app.get("/info")
def info():
    m = engine.model
    return {"params": m.num_params(), "config": m.config_dict(), "device": engine.device,
            "iter": engine.ckpt.get("iter"), "best_val_loss": engine.ckpt.get("best_val")}


@app.post("/generate")
def generate(req: GenRequest):
    """Streams the story as plain text, token by token."""
    def gen():
        with lock:
            yield from engine.stream(req.prompt, req.max_new_tokens, **_sampling(req))
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.post("/complete")
def complete(req: GenRequest):
    """Non-streaming variant: returns JSON."""
    with lock:
        t = time.time()
        text = engine.complete(req.prompt, req.max_new_tokens, **_sampling(req))
        return {"text": text, "seconds": round(time.time() - t, 3)}


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Mini GPT</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;line-height:1.5}
 textarea,pre{width:100%;box-sizing:border-box;font:inherit;padding:.6rem;border:1px solid #8884;border-radius:8px;background:transparent;color:inherit}
 pre{white-space:pre-wrap;min-height:10rem}
 label{display:inline-block;margin:.4rem 1rem .4rem 0;font-size:.9rem}
 button{padding:.5rem 1.2rem;border-radius:8px;border:0;background:#4f46e5;color:#fff;font-size:1rem;cursor:pointer}
 small{opacity:.7}
</style>
<h1>Mini GPT</h1>
<p><small id=info>loading model info…</small></p>
<textarea id=prompt rows=2>Once upon a time</textarea>
<div>
 <label>temperature <input id=temperature type=range min=0 max=1.5 step=.05 value=.8> <span id=tv>0.8</span></label>
 <label>top-p <input id=top_p type=range min=.1 max=1 step=.05 value=.95> <span id=pv>0.95</span></label>
 <label>max tokens <input id=max_new_tokens type=number min=10 max=1000 value=200 style=width:5rem></label>
</div>
<p><button id=go>Generate</button> <small id=stats></small></p>
<pre id=out></pre>
<script>
const $=id=>document.getElementById(id);
$('temperature').oninput=e=>$('tv').textContent=e.target.value;
$('top_p').oninput=e=>$('pv').textContent=e.target.value;
fetch('/info').then(r=>r.json()).then(i=>$('info').textContent=
  `${(i.params/1e6).toFixed(1)}M parameters · ${i.device} · trained ${i.iter} steps · val loss ${i.best_val_loss?.toFixed(3)}`);
$('go').onclick=async()=>{
  $('out').textContent=$('prompt').value;$('go').disabled=true;const t0=performance.now();let chars=0;
  const body={prompt:$('prompt').value,temperature:+$('temperature').value,top_p:+$('top_p').value,max_new_tokens:+$('max_new_tokens').value};
  const res=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const reader=res.body.getReader(),dec=new TextDecoder();
  for(;;){const {done,value}=await reader.read();if(done)break;
    const s=dec.decode(value,{stream:true});chars+=s.length;$('out').textContent+=s;
    $('stats').textContent=`${(chars/((performance.now()-t0)/1000)).toFixed(0)} chars/s`;}
  $('go').disabled=false;
};
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
