import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QHBoxLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from gamedataclasses import PhaseRoleMatrix, PhaseRoleTasks, PayoffConsequence
import asyncio

class ParserThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    def __init__(self, parser, feedback_prompt=None):
        super().__init__()
        self.parser = parser
        self.feedback_prompt = feedback_prompt
    def run(self):
        try:
            if self.feedback_prompt:
                messages = [
                    {"role": "system", "content": "You are a JSON extractor."},
                    {"role": "user", "content": self.feedback_prompt}
                ]
                tracing_extra = {}
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                content = loop.run_until_complete(self.parser.llm.get_response(messages, tracing_extra))
                self.parser.handle_response(content)
            else:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.parser.parse())
            self.finished.emit(self.parser)
        except Exception as e:
            self.error.emit(str(e))

class PhaseMatrixGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phase Role Matrix Viewer")
        self.setGeometry(100, 100, 900, 800)
        self.layout = QVBoxLayout()

        # Parser state
        self.state_label = QLabel("State: Idle")
        self.layout.addWidget(self.state_label)

        # Game selection
        self.layout.addWidget(QLabel("Select Game Spec:"))
        self.game_selector = QComboBox()
        self.layout.addWidget(self.game_selector)

        # Run and retry buttons
        btn_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Parser")
        self.run_button.clicked.connect(self.run_parser)
        btn_layout.addWidget(self.run_button)
        self.retry_button = QPushButton("Retry with Feedback Prompt")
        self.retry_button.clicked.connect(self.retry_with_feedback)
        btn_layout.addWidget(self.retry_button)
        self.layout.addLayout(btn_layout)

        # Status and error
        self.status_label = QLabel("Status: Waiting for input...")
        self.layout.addWidget(self.status_label)
        self.error_label = QLabel("")
        self.layout.addWidget(self.error_label)

        # Matrix and payoff tables
        self.matrix_table = QTableWidget()
        self.layout.addWidget(self.matrix_table)
        self.layout.addWidget(QLabel("Payoff Consequences:"))
        self.payoff_table = QTableWidget()
        self.layout.addWidget(self.payoff_table)

        # Editable feedback prompt
        self.layout.addWidget(QLabel("Feedback Prompt to LLM (editable):"))
        self.feedback_output = QTextEdit()
        self.feedback_output.setReadOnly(False)
        self.layout.addWidget(self.feedback_output)

        self.setLayout(self.layout)

        self.parser = None
        self.selected_game = None
        self._init_parser_and_games()
        self._reset_gui()

    def _init_parser_and_games(self):
        from parser import GameSpecParser
        self.parser = GameSpecParser()
        self.game_selector.clear()
        self.game_selector.addItems(self.parser.available_games)
        if self.parser.available_games:
            self.selected_game = self.parser.available_games[0]
            self.game_selector.setCurrentText(self.selected_game)
        self.game_selector.currentTextChanged.connect(self._on_game_selected)

    def _on_game_selected(self, game_name):
        self.selected_game = game_name
        self.parser.start(game_filename=game_name)
        self._reset_gui()

    def run_parser(self):
        self.show_status("Waiting for LLM response...")
        self.state_label.setText("State: Parsing")
        self.error_label.setText("")
        self.parser.start(game_filename=self.selected_game)
        self.thread = ParserThread(self.parser)
        self.thread.finished.connect(self._on_parser_finished)
        self.thread.error.connect(self._on_parser_error)
        self.thread.start()

    def retry_with_feedback(self):
        if not self.parser:
            self.show_status("No parser instance available.")
            return
        # Get human feedback from the editable box
        human_feedback = self.feedback_output.toPlainText()
        # Compose combined feedback (original prompt + human/auto feedback)
        combined_feedback = self.parser.get_combined_feedback(human_feedback=human_feedback)
        self.show_status("Retrying with feedback prompt...")
        self.state_label.setText("State: Parsing (Retry)")
        self.error_label.setText("")
        self.thread = ParserThread(self.parser, feedback_prompt=combined_feedback)
        self.thread.finished.connect(self._on_parser_finished)
        self.thread.error.connect(self._on_parser_error)
        self.thread.start()

    def _on_parser_finished(self, parser):
        self.parser = parser
        self.update_gui_from_parser()

    def _on_parser_error(self, error_msg):
        self.show_status(f"Error during parsing: {error_msg}")
        self.state_label.setText("State: Error")
        self.error_label.setText(error_msg)

    def update_gui_from_parser(self):
        # Show parser state
        self.state_label.setText(f"State: {self.parser.state.name}")
        # Show status and error
        if self.parser.last_error:
            self.error_label.setText(f"Error: {self.parser.last_error}")
        else:
            self.error_label.setText("")
        if self.parser.result:
            self.show_matrix(self.parser.result)
            self.show_payoff_consequences(self.parser.result)
            self.show_status("Parser completed successfully.")
        else:
            self.show_matrix(PhaseRoleMatrix(phases=[]))
            self.show_payoff_consequences(PhaseRoleMatrix(phases=[], payoff_consequences=[]))
            if self.parser.state == self.parser.state.ERROR:
                self.show_status("Parser failed or returned no data.")
            else:
                self.show_status("Waiting for input...")
        # Show feedback prompt (editable)
        if self.parser.last_feedback_prompt:
            self.feedback_output.setPlainText(self.parser.last_feedback_prompt)
        else:
            self.feedback_output.setPlainText("")

    def _reset_gui(self):
        self.state_label.setText("State: Idle")
        self.status_label.setText("Status: Waiting for input...")
        self.error_label.setText("")
        self.show_matrix(PhaseRoleMatrix(phases=[]))
        self.show_payoff_consequences(PhaseRoleMatrix(phases=[], payoff_consequences=[]))
        self.feedback_output.setPlainText("")

    def show_matrix(self, matrix: PhaseRoleMatrix):
        all_roles = set()
        for phase in matrix.phases:
            if phase.role_tasks:
                all_roles.update(phase.role_tasks.keys())
        all_roles = sorted(list(all_roles))
        self.matrix_table.clear()
        self.matrix_table.setRowCount(len(matrix.phases))
        self.matrix_table.setColumnCount(len(all_roles))
        self.matrix_table.setHorizontalHeaderLabels(all_roles)
        self.matrix_table.setVerticalHeaderLabels([str(phase.phase) for phase in matrix.phases])
        for row, phase in enumerate(matrix.phases):
            for col, role in enumerate(all_roles):
                tasks = phase.role_tasks.get(role, []) if phase.role_tasks else []
                # Convert dict tasks to readable strings
                task_strs = []
                for t in tasks:
                    if isinstance(t, dict):
                        # Try to extract a 'description' field, else use str(t)
                        desc = t.get('description') if hasattr(t, 'get') else None
                        task_strs.append(desc if desc else str(t))
                    else:
                        task_strs.append(str(t))
                cell_text = '\n'.join(task_strs) if task_strs else ""
                item = QTableWidgetItem(cell_text)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.matrix_table.setItem(row, col, item)
        self.matrix_table.resizeColumnsToContents()
        self.matrix_table.resizeRowsToContents()
        self.matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.matrix_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def show_payoff_consequences(self, matrix: PhaseRoleMatrix):
        payoff_list = matrix.payoff_consequences if hasattr(matrix, 'payoff_consequences') else []
        self.payoff_table.clear()
        self.payoff_table.setRowCount(len(payoff_list))
        self.payoff_table.setColumnCount(4)
        self.payoff_table.setHorizontalHeaderLabels(["Phase", "Role", "Choice", "Payoff"])
        for row, pc in enumerate(payoff_list):
            self.payoff_table.setItem(row, 0, QTableWidgetItem(str(pc.phase)))
            self.payoff_table.setItem(row, 1, QTableWidgetItem(str(pc.role)))
            self.payoff_table.setItem(row, 2, QTableWidgetItem(str(pc.choice)))
            self.payoff_table.setItem(row, 3, QTableWidgetItem(str(pc.payoff)))
        self.payoff_table.resizeColumnsToContents()
        self.payoff_table.resizeRowsToContents()
        self.payoff_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.payoff_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def show_status(self, status: str):
        self.status_label.setText(f"Status: {status}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PhaseMatrixGUI()
    gui.show()
    sys.exit(app.exec_())
