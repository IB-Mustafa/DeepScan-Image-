"""
dashboard.py
------------
Renders the Dashboard tab.
Reads live stats from history_manager — no hardcoded numbers.
"""

import customtkinter as ctk
from history_manager import get_stats, get_all_records


class DashboardView:
    def __init__(self, container: ctk.CTkFrame, navigate_to_analyze):
        self.container = container
        self.navigate_to_analyze = navigate_to_analyze

    def render(self):
        stats = get_stats()
        recent = get_all_records()[:3]  # last 3 scans

        # ── Hero ─────────────────────────────────────────────────────────────
        hero = ctk.CTkFrame(
            self.container, fg_color="#0d1117",
            corner_radius=15, border_width=1, border_color="#1e293b"
        )
        hero.pack(fill="x", pady=(0, 30))

        inner = ctk.CTkFrame(hero, fg_color="transparent")
        inner.pack(fill="both", padx=40, pady=40)

        ctk.CTkLabel(
            inner, text="✨ AI-POWERED DETECTION",
            text_color="#00f2ff", font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w")

        ctk.CTkLabel(
            inner,
            text="Detect deepfakes & AI-\ngenerated content",
            font=ctk.CTkFont(size=42, weight="bold"),
            justify="left"
        ).pack(anchor="w", pady=10)

        ctk.CTkButton(
            inner, text="Analyze an Image →",
            fg_color="#00f2ff", text_color="black",
            corner_radius=8, height=45, width=200,
            font=ctk.CTkFont(weight="bold"),
            command=self.navigate_to_analyze
        ).pack(anchor="w", pady=(20, 0))

        # ── Stats Cards ──────────────────────────────────────────────────────
        stats_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        stats_frame.pack(fill="x", pady=(0, 30))

        stats_data = [
            ("TOTAL SCANS",  str(stats["total"])),
            ("AI DETECTED",  str(stats["ai_detected"])),
            ("SENSITIVE",    str(stats["sensitive"])),
            ("CLEAN",        str(stats["clean"])),
        ]

        for title, val in stats_data:
            card = ctk.CTkFrame(
                stats_frame, fg_color="#0d1117",
                corner_radius=12, border_width=1, border_color="#1e293b"
            )
            card.pack(side="left", expand=True, fill="both", padx=5)
            ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=20)).pack(anchor="e", padx=15, pady=(10, 0))
            ctk.CTkLabel(card, text=val, font=ctk.CTkFont(size=32, weight="bold")).pack(anchor="w", padx=20)
            ctk.CTkLabel(
                card, text=title,
                text_color="#64748b", font=ctk.CTkFont(size=11, weight="bold")
            ).pack(anchor="w", padx=20, pady=(0, 20))

        # ── Recent Analyses ───────────────────────────────────────────────────
        ctk.CTkLabel(
            self.container, text="Recent Analyses",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(anchor="w", pady=(10, 15))

        recent_grid = ctk.CTkFrame(self.container, fg_color="transparent")
        recent_grid.pack(fill="x")

        if not recent:
            ctk.CTkLabel(
                recent_grid,
                text="No scans yet. Upload an image to get started.",
                text_color="#64748b"
            ).pack(pady=20)
            return

        for record in recent:
            card = ctk.CTkFrame(
                recent_grid, fg_color="#0d1117",
                corner_radius=12, border_width=1, border_color="#1e293b", height=180
            )
            card.pack(side="left", padx=5, expand=True, fill="both")
            card.pack_propagate(False)

            # Try to show actual thumbnail
            if record.get("path"):
                try:
                    from PIL import Image
                    img = Image.open(record["path"])
                    img.thumbnail((160, 120))
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                    ctk.CTkLabel(card, image=ctk_img, text="").pack(expand=True, pady=(10, 0))
                except Exception:
                    ctk.CTkLabel(card, text="Preview Not Available", text_color="#1e293b").pack(expand=True)
            else:
                ctk.CTkLabel(card, text="Preview Not Available", text_color="#1e293b").pack(expand=True)

            # Risk badge color
            risk = record.get("overall_risk", "Low")
            risk_color = {"Low": "#10b981", "Medium": "#fbbf24", "High": "#ef4444"}.get(risk, "#94a3b8")

            ctk.CTkLabel(card, text=record.get("name", "Unknown"), font=ctk.CTkFont(size=11)).pack(pady=(4, 0))
            ctk.CTkLabel(card, text=f"● {risk} Risk", text_color=risk_color, font=ctk.CTkFont(size=10)).pack(pady=(0, 8))