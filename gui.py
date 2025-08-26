import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
import json
from gamedataclasses import PhaseRoleMatrix, PhaseRoleTasks

class PhaseMatrixGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phase Role Matrix Viewer")
        self.setGeometry(100, 100, 700, 600)
        self.layout = QVBoxLayout()

        self.status_label = QLabel("Status: Waiting for input...")
        self.layout.addWidget(self.status_label)

        self.run_button = QPushButton("Run Parser")
        self.run_button.clicked.connect(self.run_parser)
        self.layout.addWidget(self.run_button)

        self.matrix_table = QTableWidget()
        self.layout.addWidget(self.matrix_table)

        self.feedback_output = QTextEdit()
        self.feedback_output.setReadOnly(True)
        self.layout.addWidget(QLabel("Feedback Prompt (if any):"))
        self.layout.addWidget(self.feedback_output)

        self.setLayout(self.layout)

    def run_parser(self):
        self.show_status("Waiting for LLM response...")
        QApplication.processEvents()
        try:
            from parser import run_parser_for_gui
            self.show_status("Running parser...")
            QApplication.processEvents()
            matrix, feedback = run_parser_for_gui()
            if matrix:
                self.show_matrix(matrix)
                self.show_status("Parser completed successfully.")
            else:
                self.show_matrix(PhaseRoleMatrix(phases=[]))
                self.show_status("Parser failed or returned no data.")
            if feedback:
                self.show_feedback(feedback)
            else:
                self.show_feedback("")
        except Exception as e:
            self.show_status(f"Error: {e}")
            self.show_feedback("")

    def show_matrix(self, matrix: PhaseRoleMatrix):
        # Collect all unique roles
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
                cell_text = '\n'.join(tasks) if tasks else ""
                item = QTableWidgetItem(cell_text)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.matrix_table.setItem(row, col, item)

        self.matrix_table.resizeColumnsToContents()
        self.matrix_table.resizeRowsToContents()
        self.matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.matrix_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def show_status(self, status: str):
        self.status_label.setText(f"Status: {status}")

    def show_feedback(self, feedback: str):
        self.feedback_output.setPlainText(feedback)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PhaseMatrixGUI()
    gui.show()
    sys.exit(app.exec_())
