"""
analyze.py — DeepScanAI
Renders the Analyze tab.

ROOT CAUSE OF BLANK SCREEN:
  CTkScrollableFrame uses an internal canvas that requires the parent to
  have a non-zero size at creation time. If it's built before the window
  geometry is finalised (common on first tab switch), the canvas stays 0x0
  and nothing renders — even with after() delays.

FIX:
  Replace CTkScrollableFrame with a plain tkinter Canvas + Scrollbar.
  This is geometry-independent: the canvas resizes via <Configure> binding
  and always renders correctly regardless of when it is created.

Other fixes:
  - Image blurred when max sensitivity score > 50%
  - "Why" reason shown in each sensitive card
  - Thread safety: UI updates via .after() on main thread only
  - "Analyze Another Image" only clears inner_frame children — never the canvas

Banner color logic:
  🔴 Red    — sensitive / harmful content detected
  🩷 Pink   — AI-generated only (not harmful)
  🟢 Green  — all clear / authentic
"""

import threading
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageFilter

import analyzer
import history_manager

RISK_COLOR = {
    "Low":    "#10b981",
    "Medium": "#fbbf24",
    "High":   "#ef4444",
}

CATEGORY_ICON = {
    "Unacceptable Exposure": "🔞",
    "Suggestive Content":    "⚠️",
    "PartialExposure":       "  ",
    "Blood / Injury":        "🩸",
    "Weapons":               "🔫",
    "Medical Content":       "🏥",
    "Distress / Threat":     "🚨",
}

CATEGORY_LABEL = {
    "Restricted Item":      "Weapon Detected",
    "Hemorrhage Alert":     "Blood / Injury",
    "Physical Trauma":      "Physical Injury",
    "Scene Threat":         "Threatening Scene",
    "Environmental Hazard": "Fire / Smoke",
    "Content Flag":         "Inappropriate Content",
    "Clinical Content":     "Medical Scene",
    "Scene Context":        "Scene Warning",
    "Text Risk":            "Harmful Text",
}

BG = "#060b14"


class AnalyzeView:
    def __init__(self, container: ctk.CTkFrame):
        self.container = container
        self._canvas   = None
        self._inner    = None
        self._build_layout()

    # ── Build permanent layout ONCE ──────────────────────────────────────────
    def _build_layout(self):
        ctk.CTkLabel(
            self.container,
            text="Analyze Image",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(anchor="w")

        ctk.CTkLabel(
            self.container,
            text="Upload an image to detect AI generation, deepfakes, and sensitive content.",
            text_color="#94a3b8"
        ).pack(anchor="w", pady=(0, 16))

        # Scroll host: canvas + scrollbar side by side
        scroll_host = tk.Frame(self.container, bg=BG)
        scroll_host.pack(fill="both", expand=True)
        scroll_host.grid_rowconfigure(0, weight=1)
        scroll_host.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(scroll_host, bg=BG, highlightthickness=0, bd=0)
        vsb = tk.Scrollbar(scroll_host, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)

        self._canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Inner CTkFrame placed inside canvas window
        self._inner = ctk.CTkFrame(self._canvas, fg_color="transparent")
        self._win_id = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")

        # Keep inner frame width = canvas width
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfig(  # type: ignore[union-attr]
            self._win_id, width=e.width))
        # Update scroll region whenever inner frame changes size
        self._inner.bind("<Configure>", lambda e: self._canvas.configure(  # type: ignore[union-attr]
            scrollregion=self._canvas.bbox("all")))  # type: ignore[attr-defined]

        # Mousewheel
        self._canvas.bind("<Enter>", lambda _: self._canvas.bind_all(  # type: ignore[union-attr]
            "<MouseWheel>", self._mwheel))
        self._canvas.bind("<Leave>", lambda _: self._canvas.unbind_all("<MouseWheel>"))  # type: ignore[union-attr]

        self._show_upload_prompt()

    def _mwheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")  # type: ignore[union-attr]

    # ── Clear inner frame children only ──────────────────────────────────────
    def _clear_inner(self):
        try:
            for w in list(self._inner.winfo_children()):  # type: ignore[union-attr]
                try:
                    w.destroy()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self._canvas.yview_moveto(0)  # type: ignore[union-attr]
        except Exception:
            pass

    # ── Upload prompt ─────────────────────────────────────────────────────────
    def _show_upload_prompt(self):
        self._clear_inner()

        drop = ctk.CTkFrame(
            self._inner, fg_color="#0d1117", height=260,
            corner_radius=15, border_width=1, border_color="#1e293b"
        )
        drop.pack(fill="x", pady=(0, 20), padx=2)
        drop.pack_propagate(False)

        ctk.CTkLabel(drop, text="📁", font=ctk.CTkFont(size=52)).place(
            relx=0.5, rely=0.33, anchor="center")

        ctk.CTkButton(
            drop,
            text="Click to browse  —  JPG · PNG · WEBP",
            fg_color="transparent",
            border_width=1, border_color="#00f2ff",
            text_color="#00f2ff", hover_color="#051e2b",
            height=50, command=self._pick_file
        ).place(relx=0.5, rely=0.65, anchor="center")

    # ── File picker ───────────────────────────────────────────────────────────
    def _pick_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        if not path:
            return

        self._clear_inner()
        self._add_preview(path, blur=False)

        loading = ctk.CTkLabel(
            self._inner,
            text="⏳  Analyzing…  This may take a moment.",
            text_color="#94a3b8", font=ctk.CTkFont(size=14)
        )
        loading.pack(pady=40)

        def _worker():
            try:
                result = analyzer.analyze_image(path)
            except Exception as e:
                result = {
                    "deepfake":  {"ai_score": None, "deepfake_score": None,
                                  "reasoning": str(e), "signal_scores": {}},
                    "sensitive": {}, "overall_risk": "Low", "authentic": True,
                }
            try:
                self._canvas.after(0, lambda: self._show_results(path, result))  # type: ignore[union-attr]
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    # ── Preview ───────────────────────────────────────────────────────────────
    def _add_preview(self, path: str, blur: bool):
        frame = ctk.CTkFrame(
            self._inner, fg_color="#0d1117", corner_radius=15,
            border_width=1, border_color="#1e293b"
        )
        frame.pack(fill="x", pady=(0, 14), padx=2)
        try:
            pil_img = Image.open(path).convert("RGB")
            pil_img.thumbnail((860, 360))
            if blur:
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=18))
                ctk.CTkLabel(
                    frame,
                    text="🔒  Sensitive content — preview blurred",
                    text_color="#ef4444", font=ctk.CTkFont(size=11, weight="bold")
                ).pack(pady=(8, 0))
            ctki = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            ctk.CTkLabel(frame, image=ctki, text="").pack(pady=(8, 10))
        except Exception:
            ctk.CTkLabel(frame, text="Preview unavailable",
                         text_color="#64748b").pack(pady=30)

    # ── Results ───────────────────────────────────────────────────────────────
    def _show_results(self, path: str, results: dict):
        try:
            if not self._canvas.winfo_exists():  # type: ignore[union-attr]
                return
        except Exception:
            return

        self._clear_inner()

        risk       = results.get("overall_risk", "Low")
        deepfake   = results.get("deepfake", {})
        sensitive  = results.get("sensitive", {})
        risk_color = RISK_COLOR.get(risk, "#94a3b8")

        sensitive_max = max((v.get("score", 0) for v in sensitive.values()), default=0.0)
        should_blur   = sensitive_max > 50.0

        try:
            history_manager.add_record(path, results)
        except Exception:
            pass

        self._add_preview(path, blur=should_blur)

        # ── Banner ────────────────────────────────────────────────────────────
        reasoning = deepfake.get("reasoning", "")
        ai_score  = deepfake.get("ai_score") or 0
        has_ai    = ai_score >= 45
        has_sens  = bool(sensitive)

        if has_sens:
            # 🔴 Red — actual sensitive / harmful content (always takes priority)
            bg, bdr  = "#1a0a0a", "#ef4444"
            msg      = "⚠️  Sensitive content detected"
            risk_bg  = "#2d0d0d"
            parts    = []
            if has_ai:
                parts.append(f"AI generation probability: {ai_score:.0f}%")
            flag_names = [CATEGORY_LABEL.get(k, k) for k in sensitive.keys()]
            parts.append("Found: " + ", ".join(flag_names)) # type: ignore[list-item]
            if reasoning:
                parts.append(reasoning)
            desc = " | ".join(parts)

        elif has_ai:
            # 🩷 Pink — AI-generated but no harmful content
            bg, bdr  = "#1a0a1a", "#e879f9"
            msg      = "🤖  AI-generated image detected"
            risk_bg  = "#2d0a2d"
            desc     = reasoning or f"Primary signal: AI generation probability {ai_score:.0f}%"

        else:
            # 🟢 Green — authentic / all clear
            bg, bdr  = "#061a14", "#10b981"
            msg      = "🛡️  Image appears authentic"
            risk_bg  = "#0d2d24"
            desc     = reasoning or "No AI, deepfake, or sensitive content indicators found."

        banner = ctk.CTkFrame(self._inner, fg_color=bg,
                              border_width=1, border_color=bdr, corner_radius=12)
        banner.pack(fill="x", pady=(0, 14), padx=2)

        hrow = ctk.CTkFrame(banner, fg_color="transparent")
        hrow.pack(fill="x", padx=20, pady=(14, 4))
        ctk.CTkLabel(hrow, text=msg, text_color=bdr,
                     font=ctk.CTkFont(size=17, weight="bold")).pack(side="left")
        ctk.CTkLabel(hrow, text=f"● {risk} Risk", text_color=risk_color,
                     fg_color=risk_bg, corner_radius=10,
                     font=ctk.CTkFont(size=11, weight="bold"),
                     width=90).pack(side="left", padx=12)

        ctk.CTkLabel(banner, text=desc, text_color="#94a3b8",
                     justify="left", wraplength=820).pack(
            padx=20, pady=(0, 14), anchor="w")

        # ── Probability cards ─────────────────────────────────────────────────
        cards_row = ctk.CTkFrame(self._inner, fg_color="transparent")
        cards_row.pack(fill="x", pady=8, padx=2)
        self._prob_card(cards_row, "🤖 AI Generation",
                        deepfake.get("ai_score"),
                        reasoning or "Probability this image was AI-generated", "left")
        self._prob_card(cards_row, "🧬 Deepfake Detection",
                        deepfake.get("deepfake_score"),
                        "Face manipulation probability", "right")

        # ── Signal breakdown ──────────────────────────────────────────────────
        sig = deepfake.get("signal_scores", {})
        if sig:
            sf = ctk.CTkFrame(self._inner, fg_color="#0a0f18",
                              border_width=1, border_color="#1e293b", corner_radius=10)
            sf.pack(fill="x", pady=(8, 0), padx=2)
            ctk.CTkLabel(sf, text="🔬 Signal Breakdown",
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color="#64748b").pack(anchor="w", padx=16, pady=(10, 4))
            sr = ctk.CTkFrame(sf, fg_color="transparent")
            sr.pack(fill="x", padx=16, pady=(0, 10))
            for key, label in [("exif","📋 Camera Info"), ("fft","📡 Frequency"),
                                ("ela","💾 Compression"), ("noise","🔊 Noise"),
                                ("ca","🌈 Lens Effect")]:
                val = sig.get(key, 0)
                col = "#ef4444" if val >= 65 else "#fbbf24" if val >= 45 else "#10b981"
                ctk.CTkLabel(sr, text=f"{label}\n{val:.0f}%", text_color=col,
                             font=ctk.CTkFont(size=10), justify="center",
                             width=90).pack(side="left", expand=True)

        # ── Sensitive content ─────────────────────────────────────────────────
        if sensitive:
            ctk.CTkLabel(self._inner, text="⚠️  Sensitive Content Detected",
                         font=ctk.CTkFont(size=16, weight="bold"),
                         text_color="#ef4444").pack(anchor="w", pady=(18, 8), padx=2)
            items = list(sensitive.items())
            for i in range(0, len(items), 2):
                row = ctk.CTkFrame(self._inner, fg_color="transparent")
                row.pack(fill="x", pady=3, padx=2)
                for j in range(2):
                    if i + j < len(items):
                        cat, info = items[i + j]
                        display_name = CATEGORY_LABEL.get(cat, cat)
                        self._sensitive_card(row,
                            f"{CATEGORY_ICON.get(cat, '⚠️')} {display_name}",
                            info.get("subcategory", ""),
                            info.get("score", 0.0),
                            info.get("reason", ""))
                    else:
                        ctk.CTkFrame(row, fg_color="transparent").pack(
                            side="left", expand=True, fill="both", padx=5)
        else:
            ctk.CTkLabel(self._inner, text="✅  No sensitive content detected",
                         text_color="#10b981",
                         font=ctk.CTkFont(size=13)).pack(anchor="w", pady=(14, 0), padx=2)

        # ── Analyze another ───────────────────────────────────────────────────
        ctk.CTkButton(
            self._inner,
            text="🔄  Analyze Another Image",
            fg_color="transparent",
            border_width=1, border_color="#1e293b",
            text_color="#94a3b8", height=45,
            command=self._show_upload_prompt
        ).pack(fill="x", pady=20, padx=2)

    # ── Card helpers ──────────────────────────────────────────────────────────
    def _prob_card(self, master, title, score, desc, side):
        card = ctk.CTkFrame(master, fg_color="#0d1117",
                            border_width=1, border_color="#1e293b", corner_radius=12)
        card.pack(side=side, expand=True, fill="both", padx=5)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(weight="bold"),
                     text_color="#00f2ff").pack(anchor="w", padx=20, pady=(14, 5))
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=20)
        if score is not None:
            pct_str  = f"{score:.1f}%"
            pct_norm = min(score / 100.0, 1.0)
            bar_col  = "#ef4444" if score >= 70 else "#fbbf24" if score >= 45 else "#00f2ff"
        else:
            pct_str, pct_norm, bar_col = "N/A", 0.0, "#1e293b"
        pb = ctk.CTkProgressBar(row, fg_color="#1e293b", progress_color=bar_col, height=8)
        pb.set(pct_norm)
        pb.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(row, text=pct_str,
                     font=ctk.CTkFont(weight="bold")).pack(side="right", padx=(10, 0))
        ctk.CTkLabel(card, text=desc, text_color="#64748b",
                     font=ctk.CTkFont(size=12), wraplength=380,
                     justify="left").pack(anchor="w", padx=20, pady=(8, 14))

    def _sensitive_card(self, master, title, subcategory, score, reason=""):
        card = ctk.CTkFrame(master, fg_color="#0d1117",
                            border_width=1, border_color="#3d1a1a", corner_radius=12)
        card.pack(side="left", expand=True, fill="both", padx=5)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(weight="bold"),
                     text_color="#ef4444").pack(anchor="w", padx=20, pady=(14, 2))
        ctk.CTkLabel(card, text=f"Type: {subcategory}", text_color="#94a3b8",
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20)
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(8, 0))
        pb = ctk.CTkProgressBar(row, fg_color="#1e293b", progress_color="#ef4444", height=8)
        pb.set(min(score / 100.0, 1.0))
        pb.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(row, text=f"{score:.1f}%", font=ctk.CTkFont(weight="bold"),
                     text_color="#ef4444").pack(side="right", padx=(10, 0))
        if reason:
            ctk.CTkLabel(card, text=f"Why: {reason}", text_color="#64748b",
                         font=ctk.CTkFont(size=11), wraplength=340,
                         justify="left").pack(anchor="w", padx=20, pady=(6, 14))
        else:
            ctk.CTkLabel(card, text=f"Sensitivity: {score:.1f}%", text_color="#64748b",
                         font=ctk.CTkFont(size=12)).pack(anchor="w", padx=20, pady=(6, 14))