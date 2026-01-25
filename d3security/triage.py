import os
import shutil
import subprocess
import pandas as pd
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    Input,
    Select,
    Switch,
    Button,
    RichLog, 
)

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- DATA LOADING & PROCESSING ---
def load_and_score_data(filepath):
    """
    Loads data and returns a tuple: (DataFrame, is_dummy_boolean, error_message)
    """
    if not str(filepath).lower().endswith(".csv"):
        return create_dummy_data(), True, "Invalid File Type: Not a CSV"

    try:
        df = pd.read_csv(filepath)
    except Exception:
        return create_dummy_data(), True, "File Load Error: Corrupt or Unreadable"

    # --- Standard Processing for Real Data ---
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.sort_values("timestamp_utc")

    severity_weight = {
        "Informational": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4,
    }

    if "severity" in df.columns:
        df["sev_score"] = df["severity"].map(severity_weight).fillna(0)
    else:
        df["sev_score"] = 0

    df["priority_score"] = (
        df.get("sev_score", 0)
        + df.get("confidence", pd.Series([0] * len(df))).fillna(0)
        + df.get("is_external_src_ip", pd.Series([0] * len(df))).fillna(0) * 0.5
    )

    df = df.sort_values("priority_score", ascending=False)
    return df, False, None


def create_dummy_data():
    """Generates the fallback data."""
    data = {
        "event_id": [101, 102, 103, 104, 105],
        "timestamp_utc": [
            "2025-12-16 08:00:00",
            "2025-12-16 08:05:00",
            "2025-12-16 08:10:00",
            "2025-12-15 22:00:00",
            "2025-12-16 09:30:00"
        ],
        "source": ["Firewall", "EDR", "AuthLog", "Firewall", "AWS"],
        "event_type": ["Connection Denied", "Malware Detected", "Failed Login", "Port Scan", "Config Change"],
        "severity": ["Low", "Critical", "Medium", "High", "Informational"],
        "confidence": [0.8, 0.95, 0.5, 0.7, 0.9],
        "user": ["network_svc", "jdoe", "admin", "unknown", "devops"],
        "host_or_asset": ["firewall-01", "workstation-05", "server-dc", "firewall-01", "s3-bucket"],
        "src_ip": ["192.168.1.1", "10.0.0.5", "45.33.22.11", "102.33.22.11", "10.0.0.5"],
        "is_external_src_ip": [0, 0, 1, 1, 0],
        "message": ["Packet dropped", "Trojan.Win32 detected", "Bad password attempt", "Scanning detected", "Public bucket"],
        "entity": ["192.168.1.1", "workstation-05", "admin", "102.33.22.11", "s3-bucket"],
        "action": ["Block", "Quarantine", "Deny", "Block", "Log"],
        "outcome": ["Success", "Success", "Failure", "Success", "Success"],
    }
    df = pd.DataFrame(data)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["sev_score"] = df["severity"].map({"Low": 1, "Critical": 4, "Medium": 2, "High": 3, "Informational": 0})
    df["priority_score"] = (
        df["sev_score"] + df["confidence"] + df["is_external_src_ip"] * 0.5
    )
    return df.sort_values("priority_score", ascending=False)


# --- UI WIDGETS ---

class NavigableInput(Input):
    """An Input widget that yields focus when Esc is pressed."""
    
    BINDINGS = [
        Binding("escape", "return_to_table", "Focus List"),
    ]

    def action_return_to_table(self):
        self.app.query_one("VimDataTable").focus()


class YesNoScreen(Screen):
    """A modal dialog for confirmation."""
    CSS = """
    YesNoScreen {
        align: center middle;
        background: rgba(0,0,0,0.5);
    }
    #dialog {
        grid-size: 2;
        grid-gutter: 1 2;
        grid-rows: 1fr 3;
        padding: 0 1;
        width: 60;
        height: 11;
        border: thick $background 80%;
        background: $surface;
    }
    #question {
        column-span: 2;
        height: 1fr;
        content-align: center middle;
        text-style: bold;
    }
    Button { width: 100%; }
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("y", "submit_yes", "Yes"),
        Binding("n", "cancel", "No"),
    ]

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Analyze this event with Gemini AI?\n(Costs API Tokens)", id="question"),
            Button("Yes (Enter)", variant="success", id="yes"),
            Button("No (Esc)", variant="error", id="no"),
            id="dialog"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yes":
            self.dismiss(True)
        else:
            self.dismiss(False)
            
    def action_submit_yes(self):
        self.dismiss(True)
        
    def action_cancel(self):
        self.dismiss(False)


class VimDataTable(DataTable):
    """A DataTable enhanced with vim navigation."""
    
    BINDINGS = [
        Binding("h", "back", "back"),
        Binding("j", "cursor_down", "down"),
        Binding("k", "cursor_up", "up"),
        Binding("l", "select_cursor", "select"),
        Binding("enter", "select_cursor", "select"),
    ]

    def action_back(self):
        self.app.action_clear_details()

class FilterPane(Container):
    """A collapsible pane containing filter inputs."""
    
    def compose(self) -> ComposeResult:
        with Vertical(classes="filter_group"):
            yield Label("Severity:")
            yield Select(
                options=[
                    ("All", "All"),
                    ("Critical", "Critical"),
                    ("High", "High"),
                    ("Medium", "Medium"),
                    ("Low", "Low"),
                    ("Informational", "Informational")
                ],
                value="All",
                id="filter_severity"
            )
        
        with Vertical(classes="filter_group"):
            yield Label("Entity / User (Fuzzy):")
            yield Input(placeholder="e.g. jdoe", id="filter_entity")

        with Vertical(classes="filter_group"):
            yield Label("Source (Fuzzy):")
            yield Input(placeholder="e.g. Firewall", id="filter_source")
        
        with Vertical(classes="filter_group"):
            yield Label("Date (YYYY-MM-DD):")
            yield Input(placeholder="e.g. 2025-12-16", id="filter_date")

        with Horizontal(classes="filter_group_row"):
            yield Label("External IP Only? ")
            yield Switch(id="filter_external")
        
        yield Button("Clear All Filters", variant="error", id="btn_clear_filters")


class TriageDashboard(Container):
    """The Left Panel: Ranked List + Filters."""

    def compose(self) -> ComposeResult:
        yield Label(
            "XXX  USING DUMMY DATA. PRESS 'o' TO OPEN A VALID CSV. XXX",
            id="dummy_warning",
            classes="warning hidden",
        )
        yield Label("Triage Queue (High Priority First)", classes="panel_title")
        yield FilterPane(id="filter_pane", classes="hidden")
        yield VimDataTable()
        yield Label("Showing: 0 / 0", id="record_counter")

    def on_mount(self) -> None:
        table = self.query_one(VimDataTable)
        table.cursor_type = "row"
        table.add_columns("ID", "Score", "Severity", "Event Type", "Entity")


class ExplanationPanel(Static):
    current_event = reactive(None)

    def compose(self) -> ComposeResult:
        yield Label("Event Explanation & Details", classes="panel_title")
        yield Static(id="explanation_content", expand=True)

    def watch_current_event(self, event_data):
        content = self.query_one("#explanation_content")
        if event_data is None:
            content.update("Select an event to see details.")
            return

        reasons = []
        
        sev_score = event_data.get("sev_score", 0)
        if sev_score >= 4:
            reasons.append("[bold red]CRITICAL SEVERITY:[/bold red] Immediate threat to critical assets.")
        elif sev_score == 3:
            reasons.append("[bold orange]HIGH SEVERITY:[/bold orange] Significant suspicious activity.")
        
        conf = event_data.get("confidence", 0)
        if conf >= 0.8:
            reasons.append(f"[green]HIGH CONFIDENCE ({conf}):[/green] Signal is likely a True Positive.")
        elif conf <= 0.3:
            reasons.append(f"[yellow]LOW CONFIDENCE ({conf}):[/yellow] High chance of False Positive.")
        
        if event_data.get("is_external_src_ip", 0) == 1:
            reasons.append("[bold magenta]EXTERNAL SOURCE:[/bold magenta] Traffic originated from outside.")

        outcome = str(event_data.get("outcome", "")).lower()
        if outcome in ["blocked", "denied", "failure"]:
            reasons.append(f"[blue]DEFENSIVE SUCCESS:[/blue] Action was {outcome.upper()}.")

        why_text = "\n".join(reasons) if reasons else "Standard operational event. No immediate flags."

        def get(key):
            val = str(event_data.get(key, ""))
            return val if val and val.lower() != "nan" else "-"

        display_text = f"""
[b]Event ID:[/b] {get("event_id")}  |  [b]Type:[/b] {get("event_type")}

[u][b]AUTOMATED TRIAGE:[/b][/u]
{why_text}

[u][b]WHO & WHERE:[/b][/u]
[b]User:[/b]        {get("user")}
[b]Host/Asset:[/b]  {get("host_or_asset")}
[b]Source IP:[/b]   {get("src_ip")} {"[red](EXT)[/]" if event_data.get("is_external_src_ip") == 1 else ""}
[b]Dest IP:[/b]     {get("dest_ip")}
[b]Domain:[/b]      {get("dest_domain")}

[u][b]WHAT HAPPENED:[/b][/u]
[b]Action:[/b]      {get("action")}
[b]Outcome:[/b]     {get("outcome")}
[b]Source:[/b]      {get("source")}
[b]Timestamp:[/b]   {get("timestamp_utc")}

[u][b]RAW MESSAGE:[/b][/u]
[i]{get("message")}[/i]
        """
        content.update(display_text)


class AnalystAI(Container):
    """Gemini-powered Analyst Assistant."""
    
    current_event = reactive(None)

    def compose(self) -> ComposeResult:
        yield Label("Analyst AI (Gemini 2.0 Flash)", classes="panel_title")
        yield RichLog(id="ai_chat_log", wrap=True, markup=True)
        yield NavigableInput(placeholder="Ask Gemini...", id="ai_prompt_input")

    def on_mount(self):
        self.client = None
        self.chat_session = None
        self.log_widget = self.query_one("#ai_chat_log")
        
        if not HAS_GENAI:
            self.log_widget.write("[bold red]Error:[/ red] 'google-genai' library not found.\nRun: pip install google-genai")
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                self.log_widget.write("[italic green]Gemini Client Initialized.[/]")
            except Exception as e:
                self.log_widget.write(f"[bold red]Client Init Error:[/]\n{e}")
        else:
            self.log_widget.write("[bold yellow]Warning:[/ yellow] GEMINI_API_KEY env var not set.")

    def init_context_with_dataframe(self, df):
        """Called when a CSV is loaded. Feeds context to Gemini."""
        if not self.client: return
        
        context_str = df.to_csv(index=False)
        
        sys_instruct = (
            "You are an expert SOC Analyst Assistant. You have access to the full triage dataset below. "
            "Your job is to explain events, suggest investigative steps, and identify anomalies. Be concise, professional, and explainable (cite exact events to produce summary).\n\n"
            f"DATASET CONTEXT:\n{context_str}"
        )
        try:
            self.chat_session = self.client.chats.create(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(system_instruction=sys_instruct)
            )
            self.log_widget.write(f"\n[bold blue]System:[/ bold blue] Loaded full dataset ({len(df)} rows).")
        except Exception as e:
            self.log_widget.write(f"[bold red]Chat Init Error:[/]\n{e}")

    async def on_input_submitted(self, event: Input.Submitted):
        """Handle user typing a prompt manually."""
        user_msg = event.value
        self.query_one("#ai_prompt_input").value = "" 
        
        if not self.chat_session:
            self.log_widget.write("[red]Chat not initialized (Load a file first).[/]")
            return

        self.log_widget.write(f"\n[bold cyan]You:[/ bold cyan] {user_msg}")
        self.send_to_gemini(user_msg)

    @work
    async def analyze_current_selection(self):
        """Triggered explicitly by the user via the Modal."""
        event_data = self.current_event
        if not event_data or not self.chat_session:
            return

        row_id = event_data.get('event_id', 'Unknown')
        prompt = f"Briefly analyze Event {row_id}: {event_data.get('message', '')} (Severity: {event_data.get('severity')})"
        
        self.log_widget.write(f"\n[dim]Analyzing Event {row_id}...[/dim]")
        self.send_to_gemini(prompt)

    @work(exclusive=True) 
    async def send_to_gemini(self, prompt):
        try:
            response = self.chat_session.send_message(prompt)
            self.log_widget.write(f"[bold purple]Gemini:[/ bold purple] {response.text}")
        except Exception as e:
            self.log_widget.write(f"[bold red]API Error:[/]\n{e}")


# --- MAIN APPLICATION ---
class SOCTriageApp(App):
    CSS = """
    Screen { layout: horizontal; }
    
    #left_pane { width: 40%; height: 100%; border-right: solid $primary; }
    #right_pane { width: 60%; height: 100%; layout: vertical; }
    
    TriageDashboard { height: 100%; layout: vertical; }
    
    #filter_pane {
        height: auto;
        max-height: 35; 
        background: $surface-darken-1;
        border-bottom: solid $primary;
        padding: 1;
        layout: vertical;
    }
    
    .filter_group { height: auto; margin-bottom: 1; }
    .filter_group_row { height: auto; align-vertical: middle; margin-bottom: 1; }
    #btn_clear_filters { width: 100%; }
    
    ExplanationPanel { 
        height: 45%; /* Adjusted Height */
        border-bottom: solid $secondary; 
        padding: 0; /* Remove padding to align title */
        margin: 0;
    }
    
    AnalystAI { 
        height: 55%; /* Adjusted Height */
        padding: 0; /* Remove padding to align title */
        layout: vertical; 
        margin: 0;
    }
    
    #explanation_content { padding: 0 1; }
    #ai_chat_log { 
        height: 1fr; 
        border: solid $primary; 
        background: $surface; 
        margin: 0 1;
    }
    #ai_prompt_input { 
        height: 3; 
        dock: bottom; 
        margin: 1 0 0 0;
    }

    .panel_title { background: $accent; color: auto; padding: 0 1; text-style: bold; width: 100%; }
    DataTable { height: 1fr; } 
    #record_counter { width: 100%; text-align: center; background: $primary-darken-2; color: $text-muted; padding: 0 1; }
    
    .warning { background: $error; color: white; text-align: center; text-style: bold; padding: 1; width: 100%; }
    .hidden { display: none; }
    """

    BINDINGS = [
        Binding("o", "open_file", "[o]pen file"),
        Binding("f", "toggle_filters", "[f]ilter"),
        Binding("p", "focus_prompt", "[p]rompt AI"),
        Binding("q", "smart_quit", "[q]uit"),
        Binding("escape", "close_filters", "close"),
    ]

    raw_df = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Container(id="left_pane"):
                yield TriageDashboard()
            with Container(id="right_pane"):
                yield ExplanationPanel()
                yield AnalystAI()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "D3 Security Challenge: SOC Triage v1.0"
        self.call_later(self.action_open_file)

    # --- Actions ---
    def action_focus_prompt(self):
        """Focus the AI prompt input box."""
        try:
            self.query_one("#ai_prompt_input").focus()
        except:
            pass

    def action_open_file(self):
        if not shutil.which("fzf"):
            self.notify("Error: 'fzf' is not installed.", severity="error")
            return

        with self.suspend():
            try:
                cmd = 'find . -type f -name "*.csv" | fzf'
                result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
                selected_file = result.stdout.strip()
            except Exception:
                selected_file = None

        if selected_file:
            self.load_dashboard(selected_file)
        else:
            self.notify("No file selected.", severity="warning")

    def action_toggle_filters(self):
        pane = self.query_one("#filter_pane")
        if pane.has_class("hidden"):
            pane.remove_class("hidden")
            self.query_one("#filter_severity").focus()
        else:
            pane.add_class("hidden")
            self.query_one(VimDataTable).focus()

    def action_close_filters(self):
        pane = self.query_one("#filter_pane")
        if not pane.has_class("hidden"):
            self.action_toggle_filters()

    def action_smart_quit(self):
        pane = self.query_one("#filter_pane")
        if not pane.has_class("hidden"):
            self.action_toggle_filters()
        else:
            self.exit()

    def action_clear_details(self):
        self.query_one(ExplanationPanel).current_event = None
        self.query_one(AnalystAI).current_event = None

    # --- Data & Filtering Logic ---
    def load_dashboard(self, filepath):
        self.raw_df, is_dummy, error_msg = load_and_score_data(filepath)

        warning_label = self.query_one("#dummy_warning")
        if is_dummy:
            warning_label.update(f"XXX DUMMY MODE: {error_msg}. PRESS 'o' TO RETRY. XXX")
            warning_label.remove_class("hidden")
            self.notify(error_msg, severity="error", timeout=5)
        else:
            warning_label.add_class("hidden")
            self.notify(f"Successfully loaded {len(self.raw_df)} events.")
        
        self.query_one(AnalystAI).init_context_with_dataframe(self.raw_df)
        self.reset_filter_inputs()
        self.apply_filters()
        self.query_one(VimDataTable).focus()

    def reset_filter_inputs(self):
        self.query_one("#filter_severity").value = "All"
        self.query_one("#filter_entity").value = ""
        self.query_one("#filter_source").value = ""
        self.query_one("#filter_date").value = ""
        self.query_one("#filter_external").value = False

    def apply_filters(self):
        if self.raw_df is None: return
        df = self.raw_df.copy()

        sev = self.query_one("#filter_severity").value
        if sev != "All": df = df[df["severity"] == sev]

        ent = self.query_one("#filter_entity").value
        if ent:
            mask = df["entity"].astype(str).str.contains(ent, case=False) | \
                   df["user"].astype(str).str.contains(ent, case=False)
            df = df[mask]

        src = self.query_one("#filter_source").value
        if src: df = df[df["source"].astype(str).str.contains(src, case=False)]

        dt_filter = self.query_one("#filter_date").value
        if dt_filter: df = df[df["timestamp_utc"].astype(str).str.contains(dt_filter)]

        if self.query_one("#filter_external").value:
            df = df[df["is_external_src_ip"] == 1]

        self.df = df
        self.update_table()

    def update_table(self):
        table = self.query_one(VimDataTable)
        table.clear()
        
        total = len(self.raw_df) if self.raw_df is not None else 0
        filtered = len(self.df)
        self.query_one("#record_counter").update(f"Showing {filtered} / {total} Events")

        for index, row in self.df.iterrows():
            score_fmt = f"{row.get('priority_score', 0):.2f}"
            sev = row.get("severity", "Unknown")
            sev_styled = sev
            if sev == "Critical": sev_styled = f"[bold red]{sev}[/]"
            elif sev == "High": sev_styled = f"[red]{sev}[/]"
            elif sev == "Medium": sev_styled = f"[yellow]{sev}[/]"

            table.add_row(
                str(row.get("event_id", "")),
                score_fmt,
                sev_styled,
                str(row.get("event_type", "")),
                str(row.get("entity", "")),
                key=str(index),
            )
        
        if len(self.df) > 0:
            table.move_cursor(row=0)
        else:
            self.notify("No events match current filters.", severity="warning")

    # --- Event Handlers ---
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_clear_filters":
            self.reset_filter_inputs()
            self.apply_filters()
            self.notify("Filters Cleared")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "ai_prompt_input": return 
        self.apply_filters()

    def on_select_changed(self, event: Select.Changed) -> None: self.apply_filters()
    def on_switch_changed(self, event: Switch.Changed) -> None: self.apply_filters()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = event.row_key.value
        if row_key and self.raw_df is not None:
            try:
                record = self.raw_df.loc[int(row_key)].to_dict()
                
                self.query_one(ExplanationPanel).current_event = record
                
                ai_panel = self.query_one(AnalystAI)
                ai_panel.current_event = record
                
                def confirm_analysis(should_analyze: bool):
                    if should_analyze:
                        ai_panel.analyze_current_selection()
                        
                self.push_screen(YesNoScreen(), confirm_analysis)

            except KeyError:
                pass

if __name__ == "__main__":
    app = SOCTriageApp()
    app.run()