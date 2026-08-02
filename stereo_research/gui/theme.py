APP_STYLE = """
QWidget {
    background: #0B1220;
    color: #E5EDF8;
    font-family: "Microsoft YaHei UI", "Segoe UI";
    font-size: 13px;
}
QMainWindow { background: #080E19; }
QFrame#sidebar, QFrame#topBar, QFrame#metricStrip {
    background: #101A2B;
    border: 1px solid #223149;
    border-radius: 10px;
}
QFrame#videoCard {
    background: #0D1726;
    border: 1px solid #26364F;
    border-radius: 8px;
}
QLabel#viewTitle {
    background: #121F32;
    color: #AFC4DF;
    border-bottom: 1px solid #26364F;
    font-size: 12px;
    font-weight: 600;
}
QLabel#imageViewport { background: #050A12; color: #63738A; }
QLabel#brand { color: #F8FAFC; font-size: 20px; font-weight: 700; }
QLabel#brandSub { color: #7F93AC; font-size: 11px; }
QLabel#sectionTitle { color: #CAD8EA; font-weight: 600; padding-top: 4px; }
QLabel#metricValue { color: #F8FAFC; font-family: "Cascadia Mono"; font-size: 16px; font-weight: 600; }
QLabel#metricLabel { color: #7388A3; font-size: 11px; }
QLabel#statusPill {
    background: #17243A;
    border: 1px solid #2C4261;
    border-radius: 10px;
    color: #9FB5CF;
    padding: 3px 10px;
}
QPushButton {
    min-height: 34px;
    border-radius: 6px;
    border: 1px solid #31445F;
    background: #17253A;
    color: #DDE8F5;
    padding: 0 12px;
}
QPushButton:hover { background: #203451; border-color: #4B6D98; }
QPushButton:pressed { background: #142033; }
QPushButton:focus { border: 2px solid #60A5FA; }
QPushButton:disabled { color: #57677A; background: #111B2A; border-color: #243247; }
QPushButton#primaryButton {
    background: #2563EB;
    border-color: #3B82F6;
    color: white;
    font-weight: 600;
    min-height: 42px;
}
QPushButton#primaryButton:hover { background: #3475F5; }
QPushButton#dangerButton { color: #FCA5A5; border-color: #7F3540; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 32px;
    background: #0D1726;
    border: 1px solid #2A3C57;
    border-radius: 5px;
    padding: 0 8px;
    selection-background-color: #2563EB;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #60A5FA; }
QComboBox::drop-down { border: 0; width: 24px; }
QTableWidget {
    background: #0D1726;
    alternate-background-color: #101D2F;
    border: 1px solid #26364F;
    border-radius: 6px;
    gridline-color: #223149;
    selection-background-color: #1E4F93;
}
QHeaderView::section { background: #15243A; color: #9FB5CF; padding: 6px; border: 0; border-right: 1px solid #26364F; }
QToolBox::tab { background: #142238; border: 1px solid #2A3C57; border-radius: 5px; padding: 8px; color: #B8C8DC; }
QToolBox::tab:selected { color: #FFFFFF; background: #1A3150; }
QScrollArea { border: 0; }
QProgressBar { background: #0D1726; border: 1px solid #293C56; border-radius: 5px; text-align: center; min-height: 16px; }
QProgressBar::chunk { background: #2563EB; border-radius: 4px; }
QSlider::groove:horizontal { height: 4px; background: #253651; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #60A5FA; border-radius: 7px; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QStatusBar { background: #0A111D; color: #71849D; }
QSplitter::handle { background: #18253A; width: 2px; }
QScrollBar:vertical { width: 9px; background: transparent; }
QScrollBar::handle:vertical { background: #334763; border-radius: 4px; min-height: 28px; }
"""
