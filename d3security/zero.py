import pandas as pd
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import DataTable, DirectoryTree, Footer, Header, Label, Static


# --- DATA LOADING & PROCESSING ---
def load_and_score_data(filepath):
    """
    Loads data and returns a tuple: (DataFrame, is_dummy_boolean, error_message)
    """
    # 1. Enforce CSV extension check
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
        "Informational": 0,
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4,
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
        "event_id": [101, 102, 103],
        "timestamp_utc": [
            "2025-12-16 08:00:00",
            "2025-12-16 08:05:00",
            "2025-12-16 08:10:00",
        ],
        "source": ["Firewall", "EDR", "AuthLog"],
        "event_type": ["Connection Denied", "Malware Detected", "Failed Login"],
        "severity": ["Low", "Critical", "Medium"],
        "confidence": [0.8, 0.95, 0.5],
        "user": ["network_svc", "jdoe", "admin"],
        "host_or_asset": ["firewall-01", "workstation-05", "server-dc"],
        "src_ip": ["192.168.1.1", "10.0.0.5", "45.33.22.11"],
        "is_external_src_ip": [0, 0, 1],
        "message": ["Packet dropped", "Trojan.Win32 detected", "Bad password attempt"],
        "entity": ["192.168.1.1", "workstation-05", "admin"],
        "action": ["Block", "Quarantine", "Deny"],
        "outcome": ["Success", "Success", "Failure"],
    }
    df = pd.DataFrame(data)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["sev_score"] = df["severity"].map({"Low": 1, "Critical": 4, "Medium": 2})
    df["priority_score"] = (
        df["sev_score"] + df["confidence"] + df["is_external_src_ip"] * 0.5
    )
    return df.sort_values("priority_score", ascending=False)


# --- UI WIDGETS ---


class VimDataTable(DataTable):
    """A DataTable enhanced with vim navigation."""

    BINDINGS = [
        Binding("k", "cursor_up", "up"),
        Binding("j", "cursor_down", "down"),
        Binding("h", "back", "back"),  # Custom action
        Binding("l", "select_cursor", "details"),  # Triggers RowSelected
    ]

    def action_back(self):
        """Action to clear details / 'go back'."""
        # We call a method on the main app to clear the right pane
        self.app.action_clear_details()


class VimDirectoryTree(DirectoryTree):
    """A DirectoryTree with vim navigation."""

    BINDINGS = [
        Binding("j", "cursor_down", "down"),
        Binding("k", "cursor_up", "up"),
        Binding("h", "cursor_left", "parent"),
        Binding("l", "select_cursor", "select"),
    ]


class FileSelectorScreen(Screen):
    """A modal screen to select the CSV file."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(" Select your Triage CSV File:", classes="prompt")
        yield VimDirectoryTree("./")
        yield Footer()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
        self.app.load_dashboard(event.path)


class TriageDashboard(Static):
    """The Left Panel: Ranked List of Items."""

    def compose(self) -> ComposeResult:
        yield Label(
            "!!!  USING DUMMY DATA. PRESS 'o' TO OPEN A VALID CSV. !!!",
            id="dummy_warning",
            classes="warning hidden",
        )
        yield Label("Triage Queue (High Priority First)", classes="panel_title")
        yield VimDataTable()

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
        if event_data.get("sev_score", 0) >= 3:
            reasons.append(
                f"Severity is {event_data.get('severity', 'Unknown').upper()}."
            )
        if event_data.get("is_external_src_ip", 0) == 1:
            reasons.append("Source IP is EXTERNAL.")
        if event_data.get("confidence", 0) < 0.5:
            reasons.append("Confidence is low.")

        why_text = " ".join(reasons) if reasons else "Routine event."

        display_text = f"""
[b]Event ID:[/b] {event_data.get("event_id", "N/A")}
[b]Why this matters:[/b] [yellow]{why_text}[/yellow]

[b]Timestamp:[/b] {event_data.get("timestamp_utc", "N/A")}
[b]Source:[/b] {event_data.get("source", "N/A")}
[b]User:[/b] {event_data.get("user", "N/A")}
[b]Message:[/b] {event_data.get("message", "N/A")}
        """
        content.update(display_text)


class AnalystAI(Static):
    current_event = reactive(None)

    def compose(self) -> ComposeResult:
        yield Label("Analyst AI Assistant", classes="panel_title")
        yield Static(id="ai_content", expand=True)

    def watch_current_event(self, event_data):
        content = self.query_one("#ai_content")
        if event_data is None:
            content.update("Waiting for selection...")
            return

        next_steps = []
        if event_data.get("user"):
            next_steps.append(
                f"- Pivot: Check other logs for user '{event_data['user']}'"
            )
        if event_data.get("src_ip"):
            next_steps.append(f"- Intel: Check reputation of IP {event_data['src_ip']}")

        summary = f"I have analyzed event #{event_data.get('event_id', '?')}."

        display_text = f"""
[green]{summary}[/green]

[b]Suggested Next Steps:[/b]
{chr(10).join(next_steps)}
        """
        content.update(display_text)


# --- MAIN APPLICATION ---
class SOCTriageApp(App):
    CSS = """
    Screen { layout: horizontal; }
    
    FileSelectorScreen { 
        layout: vertical; 
        background: $surface;
    }
    FileSelectorScreen .prompt {
        text-align: center;
        padding: 1;
        background: $primary;
        color: white;
        text-style: bold;
    }

    #left_pane { width: 40%; height: 100%; border-right: solid $primary; }
    #right_pane { width: 60%; height: 100%; layout: vertical; }
    TriageDashboard { height: 100%; }
    ExplanationPanel { height: 40%; border-bottom: solid $secondary; padding: 1; }
    AnalystAI { height: 60%; padding: 1; }
    .panel_title { background: $accent; color: auto; padding: 0 1; text-style: bold; width: 100%; }
    DataTable { height: 100%; }
    
    .warning {
        background: $error;
        color: white;
        text-align: center;
        text-style: bold;
        padding: 1;
        width: 100%;
    }
    .hidden { display: none; }
    """

    BINDINGS = [
        Binding("o", "open_file", "[o]pen File"),
        Binding("q", "quit", "[q]uit"),
    ]

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
        self.title = "D3 Security Challenge: SOC Triage v0.1"
        self.push_screen(FileSelectorScreen())

    def action_open_file(self):
        self.push_screen(FileSelectorScreen())

    def action_clear_details(self):
        """Custom action called by 'h' in DataTable to clear the view."""
        self.query_one(ExplanationPanel).current_event = None
        self.query_one(AnalystAI).current_event = None

    def load_dashboard(self, filepath):
        self.pop_screen()
        self.df, is_dummy, error_msg = load_and_score_data(filepath)

        warning_label = self.query_one("#dummy_warning")
        if is_dummy:
            warning_label.update(
                f"!!!  DUMMY MODE: {error_msg.upper()}. PRESS 'o' TO RETRY. !!!"
            )
            warning_label.remove_class("hidden")
            self.notify(error_msg, severity="error", timeout=5)
        else:
            warning_label.add_class("hidden")
            self.notify(f"Successfully loaded {len(self.df)} events.")

        table = self.query_one(VimDataTable)
        table.clear()

        # Clear previous details when loading new file
        self.action_clear_details()

        for index, row in self.df.iterrows():
            score_fmt = f"{row.get('priority_score', 0):.2f}"
            sev = row.get("severity", "Unknown")

            sev_styled = sev
            if sev == "Critical":
                sev_styled = f"[bold red]{sev}[/]"
            elif sev == "High":
                sev_styled = f"[red]{sev}[/]"
            elif sev == "Medium":
                sev_styled = f"[yellow]{sev}[/]"

            table.add_row(
                str(row.get("event_id", "")),
                score_fmt,
                sev_styled,
                str(row.get("event_type", "")),
                str(row.get("entity", "")),
                key=str(index),
            )

        table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Called when 'l' or 'Enter' is pressed on a row."""
        row_key = event.row_key.value
        if row_key and hasattr(self, "df"):
            record = self.df.loc[int(row_key)].to_dict()
            self.query_one(ExplanationPanel).current_event = record
            self.query_one(AnalystAI).current_event = record

if __name__ == "__main__":
    app = SOCTriageApp()
    app.run()