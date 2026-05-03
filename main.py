"""
main.py — DeepScanAI
Entry point. Builds the main window with tab navigation.
"""

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DeepScanAI")
        self.geometry("1100x780")
        self.minsize(900, 650)
        self.configure(fg_color="#060b14")

        self._active_tab   = None
        self._tab_buttons  = {}
        self._content_frames = {}

        self._build_sidebar()
        self._build_content_area()
        self._switch_tab("Dashboard")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=200, fg_color="#0a0f1a", corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(
            sidebar, text="🔍 DeepScanAI",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00f2ff"
        ).pack(pady=(28, 30), padx=20, anchor="w")

        for name, icon in [("Dashboard", "🏠"), ("Analyze", "🔬"), ("History", "📋")]:
            btn = ctk.CTkButton(
                sidebar, text=f"  {icon}  {name}",
                anchor="w", fg_color="transparent",
                text_color="#94a3b8", hover_color="#1e293b",
                font=ctk.CTkFont(size=14), height=42,
                command=lambda n=name: self._switch_tab(n)
            )
            btn.pack(fill="x", padx=10, pady=3)
            self._tab_buttons[name] = btn

    # ── Content area ──────────────────────────────────────────────────────────
    def _build_content_area(self):
        self._main_area = ctk.CTkFrame(self, fg_color="#060b14", corner_radius=0)
        self._main_area.pack(side="left", fill="both", expand=True)

    def _switch_tab(self, name: str):
        if self._active_tab == name:
            return

        # Highlight active button
        for n, btn in self._tab_buttons.items():
            if n == name:
                btn.configure(fg_color="#1e293b", text_color="#ffffff")
            else:
                btn.configure(fg_color="transparent", text_color="#94a3b8")

        # Hide all existing content frames
        for frame in self._content_frames.values():
            frame.pack_forget()

        # Build frame if first visit, else re-show
        if name not in self._content_frames:
            frame = ctk.CTkFrame(self._main_area, fg_color="transparent")
            frame.pack(fill="both", expand=True, padx=30, pady=24)
            self._content_frames[name] = frame
            self._build_tab(name, frame)
        else:
            frame = self._content_frames[name]
            frame.pack(fill="both", expand=True, padx=30, pady=24)

            # Refresh History tab every time it is shown (new scans may have been added)
            if name == "History":
                for w in frame.winfo_children():
                    w.destroy()
                from history import HistoryView
                HistoryView(frame).render()

            # Refresh Dashboard tab every time it is shown
            if name == "Dashboard":
                for w in frame.winfo_children():
                    w.destroy()
                from dashboard import DashboardView
                DashboardView(frame, lambda: self._switch_tab("Analyze")).render()

        self._active_tab = name

    def _build_tab(self, name: str, frame: ctk.CTkFrame):
        if name == "Dashboard":
            from dashboard import DashboardView
            DashboardView(frame, lambda: self._switch_tab("Analyze")).render()

        elif name == "Analyze":
            from analyze import AnalyzeView
            # Defer until frame geometry is finalized by tkinter
            frame.after(10, lambda f=frame: AnalyzeView(f))

        elif name == "History":
            from history import HistoryView
            HistoryView(frame).render()


if __name__ == "__main__":
    app = App()
    app.mainloop()