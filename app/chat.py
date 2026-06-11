from __future__ import annotations

import json
import boto3
import streamlit as st
import streamlit.components.v1 as components

# ── 预设快捷问题 ──────────────────────────────────────────────────────────────
QUICK_REPLIES = [
    "📊 What cell types are in the current dataset?",
    "🔬 What are the marker genes for Cluster 0?",
    "⚙️ How do I set the Leiden resolution?",
    "📖 How do I upload my own data?",
]

# ── 数据上下文构建 ─────────────────────────────────────────────────────────────

def _get_data_context(adata) -> str:
    if adata is None:
        return "No dataset loaded."

    lines = [f"Cells: {adata.n_obs}, Genes: {adata.n_vars}"]

    col = (
        "cell_type" if "cell_type" in adata.obs.columns
        else "leiden" if "leiden" in adata.obs.columns
        else "louvain" if "louvain" in adata.obs.columns
        else None
    )
    if col:
        counts = adata.obs[col].value_counts()
        total = len(adata.obs)
        lines.append(f"\nCell composition (by '{col}'):")
        for name, cnt in counts.items():
            lines.append(f"  {name}: {cnt} cells ({cnt / total * 100:.1f}%)")

    if "rank_genes_groups" in adata.uns:
        try:
            import pandas as pd
            df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(5)
            lines.append("\nTop 5 marker genes per cluster:")
            lines.append(df.to_string())
        except Exception:
            pass
    else:
        lines.append("\nMarker genes not yet computed (click Compute in the Marker Genes tab).")

    if "leiden" in adata.obs.columns:
        lines.append(f"\nLeiden clusters: {adata.obs['leiden'].nunique()}")
    lines.append("UMAP: " + ("computed" if "X_umap" in adata.obsm else "not computed"))

    return "\n".join(lines)


def _system_prompt(adata) -> str:
    return f"""You are the AI assistant for CellPortal, specializing in single-cell RNA sequencing (scRNA-seq) analysis.

Current dataset:
{_get_data_context(adata)}

Your capabilities:
1. Data Q&A: interpret cell type composition, marker genes, and clustering results from the data above
2. Method explanations: explain scRNA-seq algorithms and parameters (UMAP, Leiden clustering, CellTypist, etc.)
3. Platform guidance: help users operate CellPortal (upload data, run analysis, view results)

Response rules (strictly follow):
- Always respond in English
- Do NOT use # or ## headings; use **bold** for emphasis instead
- Keep lists to 6 items or fewer
- Keep total response under 150 words"""


# ── Gemini API 调用 ───────────────────────────────────────────────────────────

def call_llm(user_message: str, history: list, adata) -> str:
    """调用 AWS Bedrock Claude 3.5 Haiku，返回文字回答。"""
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=st.secrets["aws"]["AWS_DEFAULT_REGION"],
        aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
    )

    messages = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": _system_prompt(adata),
        "messages": messages,
    })

    try:
        response = client.invoke_model(
            modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            body=body,
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        return f"Sorry, request failed: {e}"


# ── UI 常量 ───────────────────────────────────────────────────────────────────

_FAB_OPEN    = "💬"
_FAB_CLOSE   = "✕"
_PHONE_BTN   = "📞"
_PANEL_MARKER = "cp-panel-anchor"
_CALL_ANCHOR  = "cp-call-anchor"

_CSS = """
<style>
/* ── Chat FAB ── */
.cp-fab-wrapper {
    position: fixed !important;
    bottom: 140px !important;
    right: 26px !important;
    z-index: 10000 !important;
    width: 60px !important;
    height: 60px !important;
}

/* ── Phone FAB ── */
.cp-phone-wrapper {
    position: fixed !important;
    bottom: 66px !important;
    right: 26px !important;
    z-index: 10000 !important;
    width: 60px !important;
    height: 60px !important;
}
.cp-phone-wrapper button {
    width: 60px !important; height: 60px !important;
    border-radius: 50% !important;
    background: radial-gradient(circle at 38% 32%, #7fd4a8 0%, #2d7a54 48%, #163d2a 100%) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,.55), inset 0 1px 3px rgba(255,255,255,.30) !important;
    color: white !important; border: none !important; font-size: 26px !important;
    padding: 0 !important; cursor: pointer !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    transition: transform .15s, box-shadow .15s !important;
}
.cp-phone-wrapper button p,
.cp-phone-wrapper button div,
.cp-phone-wrapper button span {
    font-size: 26px !important; line-height: 1 !important; margin: 0 !important; padding: 0 !important;
    filter: brightness(0) invert(1) !important;
}
.cp-phone-wrapper button:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 6px 26px rgba(0,0,0,.65), inset 0 1px 3px rgba(255,255,255,.30) !important;
}
.cp-fab-wrapper button {
    width: 60px !important;
    height: 60px !important;
    border-radius: 50% !important;
    /* 渐变+内高光，仿玻璃质感 */
    background: radial-gradient(
        circle at 38% 32%,
        #7fd4a8 0%,
        #2d7a54 48%,
        #163d2a 100%
    ) !important;
    box-shadow:
        0 4px 20px rgba(0,0,0,.55),
        inset 0 1px 3px rgba(255,255,255,.30) !important;
    color: white !important;
    border: none !important;
    font-size: 28px !important;
    padding: 0 !important;
    transition: transform .15s, box-shadow .15s !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
/* Streamlit 把按钮文字包在 <p> 里，需要单独设字号 */
.cp-fab-wrapper button p,
.cp-fab-wrapper button div,
.cp-fab-wrapper button span {
    font-size: 28px !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
}
.cp-fab-wrapper button:hover {
    transform: scale(1.08) !important;
    box-shadow:
        0 6px 26px rgba(0,0,0,.65),
        inset 0 1px 3px rgba(255,255,255,.30) !important;
}

/* ── 文字聊天面板 ── */
.cp-chat-panel {
    position: fixed !important;
    bottom: 214px !important;
    right: 30px !important;
    width: 380px !important;
    max-height: 560px !important;
    overflow-y: auto !important;
    background: #111318 !important;
    border: 1px solid #2a3040 !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.6) !important;
    z-index: 9999 !important;
    padding-bottom: 8px !important;
}

/* ── 聊天内容字体大小 & 颜色 ── */
.cp-chat-panel p    { font-size: 13px !important; line-height: 1.6 !important; color: #e8eaf0 !important; }
.cp-chat-panel li   { font-size: 13px !important; line-height: 1.6 !important; color: #e8eaf0 !important; }
.cp-chat-panel span { color: #e8eaf0 !important; }
.cp-chat-panel strong, .cp-chat-panel b { color: #ffffff !important; font-weight: 600 !important; }
.cp-chat-panel h1 { font-size: 15px !important; font-weight: 700 !important; color: #ffffff !important; margin: 6px 0 4px !important; }
.cp-chat-panel h2 { font-size: 14px !important; font-weight: 600 !important; color: #ffffff !important; margin: 5px 0 3px !important; }
.cp-chat-panel h3 { font-size: 13px !important; font-weight: 600 !important; color: #ffffff !important; margin: 4px 0 2px !important; }
.cp-chat-panel code { font-size: 12px !important; color: #a8d8b0 !important; background: #1e2a22 !important; padding: 1px 4px !important; border-radius: 3px !important; }

/* ── 面板内所有普通按钮（快捷问题 + 🔄）深色风格，覆盖 Streamlit 默认白色 ── */
.cp-chat-panel button,
.cp-chat-panel button:focus,
.cp-chat-panel button:active,
.cp-chat-panel button:focus-visible {
    background: #1c2b22 !important;
    background-color: #1c2b22 !important;
    color: #c8e6d0 !important;
    border: 1px solid #2e4a38 !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    box-shadow: none !important;
    transition: background .15s, border-color .15s !important;
}
.cp-chat-panel button:hover {
    background: #26402e !important;
    background-color: #26402e !important;
    border-color: #4a7a5a !important;
    color: #e8f5ed !important;
}

/* chat input submit arrow */
.cp-chat-panel [data-testid="stChatInputSubmitButton"] button,
.cp-chat-panel [data-testid="stChatInputSubmitButton"] button:focus,
.cp-chat-panel [data-testid="stChatInputSubmitButton"] button:hover {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    color: inherit !important;
}

/* ── Voice call panel ── */
.cp-call-panel {
    position: fixed !important;
    bottom: 214px !important;
    right: 26px !important;
    width: 360px !important;
    max-height: 540px !important;
    overflow-y: auto !important;
    background: #0d1117 !important;
    border: 1px solid #1e3a28 !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.7) !important;
    z-index: 9999 !important;
    padding-bottom: 8px !important;
}
.cp-call-panel p, .cp-call-panel li { font-size: 13px !important; line-height: 1.6 !important; color: #d0e8d8 !important; }
.cp-call-panel strong, .cp-call-panel b { color: #ffffff !important; font-weight: 600 !important; }
.cp-call-panel button,
.cp-call-panel button:focus,
.cp-call-panel button:active {
    background: #1a2e20 !important;
    background-color: #1a2e20 !important;
    color: #b8d8c0 !important;
    border: 1px solid #2a4a32 !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    box-shadow: none !important;
}
.cp-call-panel button:hover {
    background: #22402a !important;
    background-color: #22402a !important;
    border-color: #3a6040 !important;
    color: #e0f0e8 !important;
}
.cp-call-panel [data-testid="stChatInputSubmitButton"] button,
.cp-call-panel [data-testid="stChatInputSubmitButton"] button:hover {
    background: transparent !important; background-color: transparent !important;
    border: none !important; color: inherit !important;
}
</style>
"""

# MutationObserver：(1) 浮动聊天面板  (2) 给 FAB 按钮加定位 class
_JS_OBSERVER = """
<script>
(function() {
    const MARKER      = "%(marker)s";
    const CALL_ANCHOR = "%(call_anchor)s";
    const FAB_LABELS  = new Set(["%(fab_open)s", "%(fab_close)s"]);
    const PHONE_BTN   = "%(phone)s";

    function stButtonWrapper(b) {
        let el = b.parentElement;
        while (el) {
            if (el.getAttribute && el.getAttribute('data-testid') === 'stButton') return el;
            el = el.parentElement;
        }
        return null;
    }

    function floatByAnchor(id, cssClass) {
        const d = window.parent.document;
        const a = d.getElementById(id);
        if (!a) return;
        let el = a;
        while (el && el.getAttribute) {
            if (el.getAttribute('data-testid') === 'stVerticalBlock') {
                if (!el.classList.contains(cssClass)) el.classList.add(cssClass);
                break;
            }
            el = el.parentElement;
        }
    }

    function applyFloat() {
        const d = window.parent.document;
        floatByAnchor(MARKER,      'cp-chat-panel');
        floatByAnchor(CALL_ANCHOR, 'cp-call-panel');

        // Style all FAB buttons
        for (const b of d.querySelectorAll('button')) {
            const txt = b.textContent.trim();
            const wrapper = stButtonWrapper(b);
            if (!wrapper) continue;

            if (FAB_LABELS.has(txt) && !b.closest('.cp-chat-panel')) {
                if (!wrapper.classList.contains('cp-fab-wrapper'))
                    wrapper.classList.add('cp-fab-wrapper');
            }
            if (txt === PHONE_BTN) {
                if (!wrapper.classList.contains('cp-phone-wrapper'))
                    wrapper.classList.add('cp-phone-wrapper');
            }
        }
    }

    applyFloat();
    const obs = new MutationObserver(applyFloat);
    obs.observe(window.parent.document.body, { childList: true, subtree: true });
})();
</script>
""" % {"marker": _PANEL_MARKER, "call_anchor": _CALL_ANCHOR,
       "fab_open": _FAB_OPEN, "fab_close": _FAB_CLOSE, "phone": _PHONE_BTN}

# Voice strip: mic button (STT) + auto-read responses (TTS)
# Runs inside the chat panel iframe; uses browser Web Speech API (Chrome recommended)
_VOICE_JS = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    background: #111318;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  body {
    display: flex; align-items: center;
    padding: 6px 14px; gap: 10px;
    border-top: 1px solid #1e2d24;
  }
  #mic {
    width: 36px; height: 36px; border-radius: 50%;
    background: #1c2b22; border: 1px solid #2e4a38;
    color: #a8d8b8; font-size: 17px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; outline: none;
    transition: background .2s, box-shadow .2s;
  }
  #mic:hover { background: #243624; }
  #mic.on {
    background: #5a1818; border-color: #c03030; color: #fff;
    animation: pulse 1.1s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(192,48,48,.5); }
    50%      { box-shadow: 0 0 0 9px rgba(192,48,48,0); }
  }
  #status {
    font-size: 11px; color: #5a8a68; flex: 1;
    overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
  }
  #tts-btn {
    width: 28px; height: 28px; border-radius: 6px;
    background: #1c2b22; border: 1px solid #2e4a38;
    color: #7ab890; font-size: 13px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; outline: none; transition: background .2s;
  }
  #tts-btn:hover { background: #243624; }
  #tts-btn.muted { color: #555; border-color: #333; }
</style>
</head>
<body>
  <button id="mic"     title="Click to speak">🎤</button>
  <span   id="status">Tap mic to speak</span>
  <button id="tts-btn" title="Toggle auto-read">🔊</button>
</body>
<script>
(function () {
  var mic    = document.getElementById('mic');
  var status = document.getElementById('status');
  var ttsBtn = document.getElementById('tts-btn');
  var pWin   = window.parent;
  var pDoc   = pWin.document;
  var pSS    = pWin.sessionStorage;

  // ── TTS toggle ───────────────────────────────────────────────────
  var ttsEnabled = pSS.getItem('cp_tts') !== 'off';
  ttsBtn.classList.toggle('muted', !ttsEnabled);
  ttsBtn.addEventListener('click', function () {
    ttsEnabled = !ttsEnabled;
    pSS.setItem('cp_tts', ttsEnabled ? 'on' : 'off');
    ttsBtn.classList.toggle('muted', !ttsEnabled);
    if (!ttsEnabled) pWin.speechSynthesis.cancel();
  });

  // ── TTS: speak new assistant messages ────────────────────────────
  function trySpeak() {
    if (!ttsEnabled) return;
    var msgs = pDoc.querySelectorAll('[data-testid="stChatMessage"]');
    var cnt  = msgs.length;
    var done = parseInt(pSS.getItem('cp_spoken') || '0');
    if (cnt <= done) return;
    var last = msgs[cnt - 1];
    if (last.querySelector('[data-testid="stChatMessageAvatarUser"]')) {
      pSS.setItem('cp_spoken', cnt); return;
    }
    var el = last.querySelector('[data-testid="stMarkdownContainer"] p');
    if (!el) return;
    var text = el.textContent.trim();
    if (!text) return;
    pSS.setItem('cp_spoken', cnt);
    pWin.speechSynthesis.cancel();
    var utt = new SpeechSynthesisUtterance(text);
    utt.lang = 'en-US'; utt.rate = 0.95; utt.pitch = 1.0;
    pWin.speechSynthesis.speak(utt);
  }
  new MutationObserver(trySpeak).observe(pDoc.body, { childList: true, subtree: true });

  // ── STT ──────────────────────────────────────────────────────────
  var SR = pWin.SpeechRecognition || pWin.webkitSpeechRecognition;
  if (!SR) {
    mic.textContent = '✗';
    status.textContent = 'Voice not supported — use Chrome';
    return;
  }

  var rec = new SR();
  rec.lang = pWin.navigator.language || 'en-US';
  rec.interimResults = false;
  rec.maxAlternatives = 1;
  var active = false;

  mic.addEventListener('click', function () {
    if (active) { rec.stop(); return; }
    try { rec.start(); }
    catch (e) { status.textContent = 'Mic error: ' + e.message; }
  });

  var AUTOSTART = "__AUTOSTART__" === "true";

  rec.onstart = function () {
    active = true;
    mic.classList.add('on');
    mic.textContent = '🔴';
    status.textContent = 'Listening…';
  };
  rec.onend = function () {
    active = false;
    mic.classList.remove('on');
    mic.textContent = '🎤';
    if (status.textContent === 'Listening…') status.textContent = 'Tap mic to speak';
  };
  rec.onerror = function (e) {
    status.textContent = e.error === 'no-speech' ? 'No speech — try again' : 'Error: ' + e.error;
  };
  rec.onresult = function (e) {
    var text = e.results[0][0].transcript;
    status.textContent = '“' + text + '”';
    submitToChat(text);
  };

  if (AUTOSTART) {
    setTimeout(function () {
      try { rec.start(); }
      catch (e) { status.textContent = 'Mic error: ' + e.message; }
    }, 500);
  }

  function submitToChat(text) {
    var ta = pDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
    if (!ta) { status.textContent = 'Input not found'; return; }
    var setter = Object.getOwnPropertyDescriptor(pWin.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(ta, text);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    setTimeout(function () {
      var sub = pDoc.querySelector('button[data-testid="stChatInputSubmitButton"]');
      if (sub) sub.click();
    }, 80);
  }
})();
</script>
</html>
"""


# ── 消息处理 ──────────────────────────────────────────────────────────────────

def _handle_message(text: str, adata_run) -> None:
    ss = st.session_state
    ss.cp_messages.append({"role": "user", "content": text})
    with st.spinner("Thinking…"):
        reply = call_llm(text, ss.cp_history, adata_run)
    ss.cp_messages.append({"role": "assistant", "content": reply})
    ss.cp_history.append({"role": "user", "content": text})
    ss.cp_history.append({"role": "assistant", "content": reply})
    st.rerun()


# Voice call component — continuous STT→LLM→TTS loop, phone-call style UI
# Python replaces __LATEST_MSG__ and __LATEST_IDX__ before passing to components.html()
_CALL_VOICE_JS = """
<!DOCTYPE html>
<html>
<head>
<style>
  html,body{margin:0;padding:0;background:#0d1117;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
  body{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:8px;gap:5px;position:relative;}
  .ring{width:68px;height:68px;border-radius:50%;background:#1a2e20;border:2px solid #2e4a38;
    font-size:28px;cursor:pointer;display:flex;align-items:center;justify-content:center;
    transition:all .25s;user-select:none;}
  .ring.listening{background:#1b3d25;border-color:#5abd78;animation:glow-g 1.2s ease-in-out infinite;}
  .ring.speaking {background:#151e30;border-color:#4a7ab8;animation:glow-b 1.4s ease-in-out infinite;}
  .ring.idle     {background:#1a2e20;border-color:#2e4a38;}
  @keyframes glow-g{0%,100%{box-shadow:0 0 0 0 rgba(90,189,120,.45);}50%{box-shadow:0 0 0 12px rgba(90,189,120,0);}}
  @keyframes glow-b{0%,100%{box-shadow:0 0 0 0 rgba(74,122,184,.45);}50%{box-shadow:0 0 0 12px rgba(74,122,184,0);}}
  #st {font-size:12px;color:#5a9a6a;letter-spacing:.3px;text-align:center;}
  #lh {font-size:12px;color:#8fcca0;max-width:310px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .hint{font-size:10px;color:#2a3a2e;}
  #lang-btn{position:absolute;top:5px;right:7px;background:#1a2e20;border:1px solid #2e4a38;
    color:#4a7a58;font-size:10px;border-radius:4px;padding:2px 6px;cursor:pointer;line-height:1.4;}
  #lang-btn:hover{background:#243624;color:#8fcca0;}
</style>
</head>
<body>
  <button id="lang-btn" title="切换识别语言 / Switch STT language">🌐 <span id="lang-lbl">--</span></button>
  <div id="ring" class="ring idle" title="Tap to start/stop">🎤</div>
  <div id="st">Starting…</div>
  <div id="lh"></div>
  <div class="hint">Tap ring to pause/resume · AI replies automatically</div>
</body>
<script>
(function(){
  var ring=document.getElementById('ring');
  var st  =document.getElementById('st');
  var lh  =document.getElementById('lh');
  var lb  =document.getElementById('lang-lbl');
  var pWin=window.parent, pDoc=pWin.document, pSS=pWin.sessionStorage;
  var active=false, starting=false, paused=false, waitingReply=false;
  var startTimer=null;

  function setSt(t){st.textContent=t;}

  // ── Language toggle (zh-CN / en-US) ──────────────────────────────
  var LANGS=['zh-CN','en-US'];
  var lang=pSS.getItem('cp_call_lang')||'en-US';
  if(!LANGS.includes(lang)) lang='en-US';
  pSS.setItem('cp_call_lang',lang); lb.textContent=lang;

  document.getElementById('lang-btn').addEventListener('click',function(){
    lang=LANGS[(LANGS.indexOf(lang)+1)%LANGS.length];
    pSS.setItem('cp_call_lang',lang); lb.textContent=lang;
    rec=buildRec();
    setSt('Language: '+lang);
  });

  // ── SpeechRecognition ─────────────────────────────────────────────
  var SR=pWin.SpeechRecognition||pWin.webkitSpeechRecognition;
  if(!SR){ring.textContent='✗';setSt('Use Chrome for voice');return;}

  function buildRec(){
    var r=new SR();
    r.lang=lang; r.interimResults=false; r.maxAlternatives=1;
    r.onstart =function(){starting=false;active=true;paused=false;ring.className='ring listening';ring.textContent='🔴';setSt('Listening… ('+lang+')');};
    r.onend   =function(){starting=false;active=false;if(ring.className!=='ring speaking'){ring.className='ring idle';ring.textContent='🎤';}if(!paused&&!waitingReply)scheduleStart(500);};
    r.onerror =function(e){starting=false;if(e.error==='aborted')return;if(e.error==='not-allowed'){ring.className='ring idle';ring.textContent='🎤';setSt('Mic blocked — allow mic in browser');return;}setSt(e.error==='no-speech'?'No speech — try again':'Mic error: '+e.error);scheduleStart(1200);};
    r.onresult=function(e){var t=e.results[0][0].transcript;lh.textContent='🗣 "'+t+'"';setSt('Waiting for AI reply…');waitingReply=true;setTimeout(function(){if(waitingReply){waitingReply=false;setSt('Timeout — tap ● to try again');}},30000);submit(t);};
    return r;
  }
  var rec=buildRec();

  function scheduleStart(d){if(startTimer)clearTimeout(startTimer);startTimer=setTimeout(startRec,d);}
  function startRec(){startTimer=null;if(active||starting||paused||waitingReply)return;starting=true;try{rec.start();}catch(e){starting=false;rec=buildRec();setSt('Resetting mic…');scheduleStart(700);}}

  ring.addEventListener('click',function(){
    if(active){rec.stop();paused=true;setSt('Paused — tap to resume');}
    else if(paused){paused=false;waitingReply=false;starting=false;rec=buildRec();setSt('Resuming…');startRec();}
    else{waitingReply=false;startRec();}
  });

  // ── Submit (identical to working _VOICE_JS approach) ─────────────
  function submit(text){
    var ta=pDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
    if(!ta){setSt('⚠ Input not found — type below');lh.textContent='';waitingReply=false;scheduleStart(1500);return;}
    var ns=Object.getOwnPropertyDescriptor(pWin.HTMLTextAreaElement.prototype,'value').set;
    ns.call(ta,text);
    ta.dispatchEvent(new Event('input',{bubbles:true}));
    setTimeout(function(){
      var sub=pDoc.querySelector('button[data-testid="stChatInputSubmitButton"]');
      if(sub) sub.click();
      else {setSt('⚠ Submit btn not found — type below');waitingReply=false;scheduleStart(1500);}
    },80);
  }

  // ── TTS: Python injects latest assistant reply via __LATEST_MSG__ ─
  // Avoids unreliable DOM scanning after Streamlit reruns.
  var injectMsg=__LATEST_MSG__;
  var injectIdx=__LATEST_IDX__;

  function speakAndResume(text){
    ring.className='ring speaking'; ring.textContent='🔊'; setSt('Speaking…');
    pWin.speechSynthesis.cancel();
    var utt=new SpeechSynthesisUtterance(text);
    utt.lang='en-US'; utt.rate=0.93; utt.pitch=1.0;
    utt.onend=function(){ring.className='ring idle';ring.textContent='🎤';if(!paused)scheduleStart(400);};
    pWin.speechSynthesis.speak(utt);
  }

  if(injectMsg && injectIdx>=0 && pSS.getItem('cp_call_sp')!==String(injectIdx)){
    pSS.setItem('cp_call_sp',String(injectIdx));
    waitingReply=false;
    speakAndResume(injectMsg);
  } else {
    setSt('Tap ● to start  ·  '+lang);
    scheduleStart(900);
  }
})();
</script>
</html>
"""


# ── 主入口 ────────────────────────────────────────────────────────────────────

def render(adata_run) -> None:
    """在页面底部注入浮动聊天组件，在 app.py 末尾调用一次即可。"""
    ss = st.session_state
    if "cp_open"      not in ss: ss.cp_open      = False
    if "cp_call_open" not in ss: ss.cp_call_open = False
    if "cp_messages"  not in ss: ss.cp_messages  = []
    if "cp_history"   not in ss: ss.cp_history   = []

    st.markdown(_CSS, unsafe_allow_html=True)

    # 💬 Chat FAB — toggles text panel; closing resets conversation
    fab_label = _FAB_CLOSE if ss.cp_open else _FAB_OPEN
    if st.button(fab_label, key="cp_fab"):
        ss.cp_open = not ss.cp_open
        if not ss.cp_open:
            ss.cp_messages = []
            ss.cp_history  = []
        ss.cp_call_open = False          # close call panel if switching to chat
        st.rerun()

    # 📞 Phone FAB — toggles voice call panel (mutually exclusive with text chat)
    if st.button(_PHONE_BTN, key="cp_phone"):
        ss.cp_call_open = not ss.cp_call_open
        if ss.cp_call_open:
            ss.cp_open = False           # hide text chat when call opens
        else:
            ss.cp_messages = []          # clear transcript when hanging up
            ss.cp_history  = []
        st.rerun()

    components.html(_JS_OBSERVER, height=0, scrolling=False)

    # ── Text chat panel ───────────────────────────────────────────────
    if ss.cp_open:
        with st.container():
            st.markdown(
                f'<div id="{_PANEL_MARKER}" style="display:none;height:0;overflow:hidden;"></div>',
                unsafe_allow_html=True,
            )

            col_title, col_new = st.columns([5, 1])
            with col_title:
                st.markdown(
                    "<div style='padding:6px 0 4px'>"
                    "<span style='font-weight:700;font-size:15px;color:#fff'>🧬 CellPortal AI Assistant</span><br>"
                    "<span style='font-size:12px;color:#888'>Answers based on your loaded data</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with col_new:
                if st.button("🔄", key="cp_new", help="New conversation"):
                    ss.cp_messages = []
                    ss.cp_history  = []
                    st.rerun()

            st.divider()

            components.html(
                _VOICE_JS.replace("__AUTOSTART__", "false"),
                height=52, scrolling=False,
            )

            for msg in ss.cp_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if not ss.cp_messages:
                st.markdown(
                    "<p style='font-size:12px;color:#aaa;margin:0 0 6px'>Try a quick question or type your own:</p>",
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns(2)
                for i, qr in enumerate(QUICK_REPLIES):
                    col = col1 if i % 2 == 0 else col2
                    if col.button(qr, key=f"cp_qr_{i}", use_container_width=True):
                        _handle_message(qr, adata_run)

            user_input = st.chat_input("Ask a question…", key="cp_input")
            if user_input:
                _handle_message(user_input, adata_run)

    # ── Voice call panel ──────────────────────────────────────────────
    if ss.cp_call_open:
        with st.container():
            st.markdown(
                f'<div id="{_CALL_ANCHOR}" style="display:none;height:0;overflow:hidden;"></div>',
                unsafe_allow_html=True,
            )

            col_title, col_hangup = st.columns([5, 1])
            with col_title:
                st.markdown(
                    "<div style='padding:6px 0 4px'>"
                    "<span style='font-weight:700;font-size:15px;color:#a0e8b0'>📞 CellPortal AI</span><br>"
                    "<span style='font-size:12px;color:#4a7a58'>Voice conversation · powered by Claude</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with col_hangup:
                if st.button("🔴", key="cp_hangup", help="End call"):
                    ss.cp_call_open = False
                    ss.cp_messages  = []
                    ss.cp_history   = []
                    st.rerun()

            st.divider()

            # Inject latest assistant message so JS speaks it without DOM scanning
            _vm, _vi = "", -1
            for _i in range(len(ss.cp_messages)-1, -1, -1):
                if ss.cp_messages[_i]["role"] == "assistant":
                    _vm, _vi = ss.cp_messages[_i]["content"], _i
                    break
            _call_js = (_CALL_VOICE_JS
                .replace("__LATEST_MSG__", json.dumps(_vm))
                .replace("__LATEST_IDX__", str(_vi)))
            components.html(_call_js, height=195, scrolling=False)

            # Transcript of the conversation
            for msg in ss.cp_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Hidden chat_input — voice JS submits here; user can also type
            call_text = st.chat_input("Or type here…", key="cp_call_input")
            if call_text:
                _handle_message(call_text, adata_run)
