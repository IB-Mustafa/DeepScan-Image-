"""
history.py
----------
Renders the History tab.
All data comes from history_manager.get_all_records() — file-backed.
"""

import customtkinter as ctk
from PIL import Image
import history_manager


RISK_COLOR = {
    "Low":    "#10b981",
    "Medium": "#fbbf24",
    "High":   "#ef4444",
}


class HistoryView:
    def __init__(self, container: ctk.CTkFrame):
        self.container = container

    def render(self):
        records = history_manager.get_all_records()

        # Header row
        header_row = ctk.CTkFrame(self.container, fg_color="transparent")
        header_row.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            header_row, text="Analysis History",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(side="left")

        if records:
            ctk.CTkButton(
                header_row,
                text="🗑  Clear All",
                fg_color="transparent",
                border_width=1, border_color="#3d1a1a",
                text_color="#ef4444",
                hover_color="#1a0808",
                width=110, height=32,
                command=self._clear_and_refresh
            ).pack(side="right")

        # Scrollable list
        scroll_frame = ctk.CTkScrollableFrame(
            self.container, fg_color="transparent", height=600
        )
        scroll_frame.pack(fill="both", expand=True)
        self._scroll_frame = scroll_frame

        if not records:
            ctk.CTkLabel(
                scroll_frame,
                text="No scan history yet.\nAnalyze an image to get started.",
                text_color="#64748b",
                font=ctk.CTkFont(size=14),
                justify="center"
            ).pack(pady=60)
            return

        # Column labels
        col_header = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        col_header.pack(fill="x", padx=10, pady=(0, 5))
        for txt, width in [("Image", 80), ("Name", 220), ("Date", 160),
                            ("Risk", 90), ("AI Score", 100), ("Deepfake", 100), ("Sensitive", 180)]:
            ctk.CTkLabel(
                col_header, text=txt, width=width,
                text_color="#64748b", font=ctk.CTkFont(size=11, weight="bold"),
                anchor="w"
            ).pack(side="left", padx=4)

        for record in records:
            self._render_row(scroll_frame, record)

    def _render_row(self, parent, record: dict):
        risk = record.get("overall_risk", "Low")
        risk_color = RISK_COLOR.get(risk, "#94a3b8")

        item = ctk.CTkFrame(
            parent, fg_color="#0d1117",
            corner_radius=10, border_width=1, border_color="#1e293b"
        )
        item.pack(fill="x", pady=4, padx=2)

        # Thumbnail
        thumb = ctk.CTkFrame(item, fg_color="#1e293b", width=50, height=50, corner_radius=5)
        thumb.pack(side="left", padx=12, pady=10)
        thumb.pack_propagate(False)

        if record.get("path"):
            try:
                img = Image.open(record["path"])
                img.thumbnail((46, 46))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                ctk.CTkLabel(thumb, image=ctk_img, text="").pack(expand=True)
            except Exception:
                ctk.CTkLabel(thumb, text="📷", font=ctk.CTkFont(size=18)).pack(expand=True)
        else:
            ctk.CTkLabel(thumb, text="📷", font=ctk.CTkFont(size=18)).pack(expand=True)

        # Name
        ctk.CTkLabel(
            item, text=record.get("name", "Unknown"),
            font=ctk.CTkFont(weight="bold"),
            width=220, anchor="w"
        ).pack(side="left", padx=4)

        # Timestamp
        ctk.CTkLabel(
            item, text=record.get("timestamp", "—"),
            text_color="#64748b",
            font=ctk.CTkFont(size=11),
            width=160, anchor="w"
        ).pack(side="left", padx=4)

        # Risk
        ctk.CTkLabel(
            item, text=f"● {risk}",
            text_color=risk_color,
            font=ctk.CTkFont(size=12, weight="bold"),
            width=90, anchor="w"
        ).pack(side="left", padx=4)

        # AI Score
        ai = record.get("ai_score")
        ctk.CTkLabel(
            item,
            text=f"{ai}%" if ai is not None else "N/A",
            text_color="#94a3b8",
            width=100, anchor="w"
        ).pack(side="left", padx=4)

        # Deepfake Score
        df = record.get("deepfake_score")
        ctk.CTkLabel(
            item,
            text=f"{df}%" if df is not None else "N/A",
            text_color="#94a3b8",
            width=100, anchor="w"
        ).pack(side="left", padx=4)

        # Sensitive categories — show subcategory only (clean language)
        sens = record.get("sensitive", {})
        if sens:
            sens_text = ", ".join(v.get("subcategory", k) for k, v in sens.items())
            color = "#ef4444"
        else:
            sens_text = "None"
            color = "#64748b"

        ctk.CTkLabel(
            item, text=sens_text,
            text_color=color,
            font=ctk.CTkFont(size=11),
            anchor="w", wraplength=200
        ).pack(side="left", padx=(4, 12))

    def _clear_and_refresh(self):
        history_manager.clear_history()
        for w in self.container.winfo_children():
            w.destroy()
        self.render()