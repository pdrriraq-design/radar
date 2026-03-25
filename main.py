import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QTimer, pyqtSignal, QThread
import pyqtgraph as pg
from rtlsdr import RtlSdr
import threading
import time
from collections import deque
import json
import os

# ==================== ثوابت الترددات العسكرية ====================
MILITARY_FREQ_BANDS = [
    (225, 260, "VHF Military Air"),
    (260, 300, "UHF Military Air"),
    (300, 380, "Military/Satcom")
]

# قاعدة بيانات الطائرات المعروفة
AIRCRAFT_DB = {
    "A-10": {
        "name": "A-10 Thunderbolt II 'Warthog'",
        "type": "Attack Aircraft",
        "country": "USA",
        "speed": "420 mph",
        "armament": "GAU-8 Avenger 30mm",
        "description": "Close air support aircraft designed around its massive cannon"
    },
    "F-16": {
        "name": "F-16 Fighting Falcon",
        "type": "Multirole Fighter",
        "country": "USA",
        "speed": "1,500 mph",
        "armament": "M61 Vulcan 20mm, AIM-9, AIM-120",
        "description": "Highly maneuverable multirole fighter"
    },
    "F-22": {
        "name": "F-22 Raptor",
        "type": "Stealth Air Superiority",
        "country": "USA",
        "speed": "1,500+ mph",
        "armament": "M61A2 Vulcan, AIM-120, AIM-9",
        "description": "5th generation stealth fighter"
    },
    "F-35": {
        "name": "F-35 Lightning II",
        "type": "Stealth Multirole",
        "country": "USA",
        "speed": "1,200 mph",
        "armament": "GAU-22/A 25mm, internal/external weapons",
        "description": "5th generation multirole stealth fighter"
    },
    "Eurofighter": {
        "name": "Eurofighter Typhoon",
        "type": "Multirole Fighter",
        "country": "Europe",
        "speed": "1,550 mph",
        "armament": "Mauser BK-27 27mm",
        "description": "European multirole fighter"
    },
    "Su-27": {
        "name": "Su-27 Flanker",
        "type": "Air Superiority",
        "country": "Russia",
        "speed": "1,550 mph",
        "armament": "GSh-30-1 30mm",
        "description": "Russian heavy air superiority fighter"
    },
    "MiG-29": {
        "name": "MiG-29 Fulcrum",
        "type": "Multirole Fighter",
        "country": "Russia",
        "speed": "1,520 mph",
        "armament": "GSh-30-1 30mm",
        "description": "Russian multirole fighter"
    },
    "C-130": {
        "name": "C-130 Hercules",
        "type": "Transport",
        "country": "USA",
        "speed": "366 mph",
        "armament": "None / Optional defensive systems",
        "description": "Tactical transport aircraft"
    },
    "E-3": {
        "name": "E-3 Sentry AWACS",
        "type": "Airborne Early Warning",
        "country": "USA",
        "speed": "460 mph",
        "armament": "None",
        "description": "Airborne Warning and Control System"
    }
}

# ==================== موضوع SDR للقراءة في خيط منفصل ====================
class SDRThread(QThread):
    """خيط منفصل لقراءة البيانات من SDR دون تعليق الواجهة"""
    data_received = pyqtSignal(np.ndarray, float, float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.sdr = None
        self.running = False
        self.freq_hz = 300e6  # تردد بدائي 300 MHz
        self.sample_rate = 2.4e6
        self.gain = 40
        
    def setup_sdr(self):
        """تهيئة جهاز SDR"""
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.freq_hz
            self.sdr.gain = self.gain
            return True
        except Exception as e:
            self.error_occurred.emit(f"فشل في تهيئة SDR: {str(e)}")
            return False
    
    def set_frequency(self, freq_hz):
        """تغيير التردد المركزي"""
        self.freq_hz = freq_hz
        if self.sdr:
            try:
                self.sdr.center_freq = freq_hz
                return True
            except:
                return False
        return True
    
    def run(self):
        """تشغيل خيط الاستقبال"""
        if not self.setup_sdr():
            return
        
        self.running = True
        
        while self.running:
            try:
                # قراءة العينات من SDR
                samples = self.sdr.read_samples(256 * 1024)
                if len(samples) > 0:
                    # تحويل العينات إلى طيف باستخدام FFT
                    fft_data = np.fft.fftshift(np.fft.fft(samples))
                    power = 20 * np.log10(np.abs(fft_data) + 1e-12)
                    
                    # حساب ترددات نقاط FFT
                    freq_range = np.linspace(
                        self.freq_hz - self.sample_rate/2,
                        self.freq_hz + self.sample_rate/2,
                        len(power)
                    )
                    
                    # حساب متوسط القدرة للكشف عن الإشارات
                    avg_power = np.mean(power)
                    max_power = np.max(power)
                    
                    self.data_received.emit(power, freq_range[0], freq_range[-1])
                    
            except Exception as e:
                self.error_occurred.emit(f"خطأ في القراءة: {str(e)}")
                time.sleep(0.1)
    
    def stop(self):
        """إيقاف الخيط وتحرير الجهاز"""
        self.running = False
        if self.sdr:
            try:
                self.sdr.close()
            except:
                pass
        self.wait()

# ==================== نافذة المعلومات التفصيلية للطائرة ====================
class AircraftInfoDialog(QtWidgets.QDialog):
    """نافذة منبثقة لعرض معلومات مفصلة عن الطائرة"""
    def __init__(self, aircraft_code, freq, power, parent=None):
        super().__init__(parent)
        self.aircraft_code = aircraft_code
        self.freq = freq
        self.power = power
        self.setWindowTitle(f"معلومات الطائرة - {aircraft_code}")
        self.setGeometry(200, 200, 450, 400)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # عنوان
        title = QtWidgets.QLabel(f"<h2>{self.aircraft_code}</h2>")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        # البحث في قاعدة البيانات
        aircraft = AIRCRAFT_DB.get(self.aircraft_code, None)
        
        if aircraft:
            # عرض الصورة (placeholder)
            pic_label = QtWidgets.QLabel()
            pic_label.setText(f"✈️ {aircraft['name']}")
            pic_label.setStyleSheet("font-size: 18px; font-weight: bold;")
            pic_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(pic_label)
            
            # جدول المعلومات
            info_widget = QtWidgets.QTableWidget()
            info_widget.setRowCount(6)
            info_widget.setColumnCount(2)
            info_widget.setHorizontalHeaderLabels(["الخاصية", "القيمة"])
            info_widget.horizontalHeader().setStretchLastSection(True)
            info_widget.verticalHeader().setVisible(False)
            
            info_data = [
                ("نوع الطائرة", aircraft.get("type", "غير معروف")),
                ("بلد المنشأ", aircraft.get("country", "غير معروف")),
                ("السرعة القصوى", aircraft.get("speed", "غير معروف")),
                ("التسليح الرئيسي", aircraft.get("armament", "غير معروف")),
                ("الوصف", aircraft.get("description", "غير معروف")),
                ("التردد المكتشف", f"{self.freq/1e6:.3f} MHz"),
                ("قوة الإشارة", f"{self.power:.1f} dB")
            ]
            
            for i, (key, val) in enumerate(info_data):
                info_widget.setItem(i, 0, QtWidgets.QTableWidgetItem(key))
                info_widget.setItem(i, 1, QtWidgets.QTableWidgetItem(val))
                info_widget.setColumnWidth(0, 120)
            
            info_widget.resizeColumnsToContents()
            layout.addWidget(info_widget)
            
        else:
            unknown_label = QtWidgets.QLabel(
                f"<p>⚠️ طائرة غير معروفة في قاعدة البيانات</p>"
                f"<p><b>الرمز:</b> {self.aircraft_code}</p>"
                f"<p><b>التردد:</b> {self.freq/1e6:.3f} MHz</p>"
                f"<p><b>قوة الإشارة:</b> {self.power:.1f} dB</p>"
            )
            unknown_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(unknown_label)
        
        # زر إغلاق
        close_btn = QtWidgets.QPushButton("إغلاق")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(close_btn)
        
        self.setLayout(layout)

# ==================== نافذة الطائرات المكتشفة ====================
class AircraftListWidget(QtWidgets.QWidget):
    """عرض قائمة الطائرات المكتشفة"""
    aircraft_selected = pyqtSignal(str, float, float)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.detected_aircraft = []
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # عنوان
        title = QtWidgets.QLabel("<h3>✈️ الطائرات المكتشفة</h3>")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        # قائمة الطائرات
        self.aircraft_list = QtWidgets.QListWidget()
        self.aircraft_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.aircraft_list)
        
        # تذييل
        self.status_label = QtWidgets.QLabel("في انتظار الاكتشاف...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
    def add_aircraft(self, code, freq, power):
        """إضافة طائرة إلى القائمة"""
        aircraft = AIRCRAFT_DB.get(code, None)
        name = aircraft["name"] if aircraft else "طائرة غير معروفة"
        
        item_text = f"{code} - {name} | {freq/1e6:.3f} MHz | {power:.1f} dB"
        self.aircraft_list.addItem(item_text)
        self.detected_aircraft.append((code, freq, power))
        self.status_label.setText(f"تم اكتشاف {len(self.detected_aircraft)} طائرة")
        
    def on_item_double_clicked(self, item):
        """عرض معلومات الطائرة عند النقر المزدوج"""
        idx = self.aircraft_list.row(item)
        code, freq, power = self.detected_aircraft[idx]
        dialog = AircraftInfoDialog(code, freq, power, self)
        dialog.exec_()

# ==================== النافذة الرئيسية للرادار ====================
class RadarWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.sdr_thread = None
        self.current_freq = 300e6
        self.sample_rate = 2.4e6
        self.spectrum_plot = None
        self.curve = None
        self.detected_peaks = []
        self.setup_ui()
        self.setup_sdr()
        
    def setup_ui(self):
        """تهيئة الواجهة الرسومية"""
        self.setWindowTitle("رادار كشف الطائرات - SDR Radar System")
        self.setGeometry(100, 100, 1400, 800)
        
        # تعيين لون داكن للواجهة
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QLabel { color: #eeeeee; }
            QPushButton { background-color: #0f3460; color: #eeeeee; border: none; padding: 8px; border-radius: 5px; }
            QPushButton:hover { background-color: #16213e; }
            QLineEdit { background-color: #16213e; color: #eeeeee; border: 1px solid #0f3460; padding: 5px; }
            QComboBox { background-color: #16213e; color: #eeeeee; border: 1px solid #0f3460; padding: 5px; }
            QListWidget { background-color: #16213e; color: #eeeeee; border: 1px solid #0f3460; }
            QTableWidget { background-color: #16213e; color: #eeeeee; grid-color: #0f3460; }
            QTableWidget::item { padding: 5px; }
            QHeaderView::section { background-color: #0f3460; color: #eeeeee; }
        """)
        
        # عنصر واجهة مركزي
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # تخطيط رئيسي
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # ===== اللوحة اليسرى (التحكم والقائمة) =====
        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # عنوان
        title = QtWidgets.QLabel("<h1>🛰️ SDR Radar System</h1>")
        title.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(title)
        
        # تحكمات التردد
        freq_group = QtWidgets.QGroupBox("🎛️ تحكم التردد")
        freq_layout = QtWidgets.QVBoxLayout()
        
        self.freq_input = QtWidgets.QLineEdit("300.0")
        self.freq_input.setPlaceholderText("التردد (MHz)")
        freq_layout.addWidget(QtWidgets.QLabel("التردد المركزي (MHz):"))
        freq_layout.addWidget(self.freq_input)
        
        # أزرار سريعة للترددات العسكرية
        freq_buttons_layout = QtWidgets.QHBoxLayout()
        for band in MILITARY_FREQ_BANDS:
            btn = QtWidgets.QPushButton(f"{band[0]}-{band[1]} MHz")
            btn.clicked.connect(lambda checked, f=band[0]: self.set_frequency(f * 1e6))
            freq_buttons_layout.addWidget(btn)
        freq_layout.addLayout(freq_buttons_layout)
        
        self.set_freq_btn = QtWidgets.QPushButton("تطبيق التردد")
        self.set_freq_btn.clicked.connect(self.apply_frequency)
        freq_layout.addWidget(self.set_freq_btn)
        
        freq_group.setLayout(freq_layout)
        left_layout.addWidget(freq_group)
        
        # تحكمات الكسب
        gain_group = QtWidgets.QGroupBox("📡 تحكم الكسب (Gain)")
        gain_layout = QtWidgets.QVBoxLayout()
        
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setMinimum(0)
        self.gain_slider.setMaximum(49)
        self.gain_slider.setValue(40)
        self.gain_slider.valueChanged.connect(self.update_gain)
        
        self.gain_label = QtWidgets.QLabel("الكسب: 40 dB")
        gain_layout.addWidget(self.gain_label)
        gain_layout.addWidget(self.gain_slider)
        
        gain_group.setLayout(gain_layout)
        left_layout.addWidget(gain_group)
        
        # حالة الاتصال
        status_group = QtWidgets.QGroupBox("🔌 حالة الجهاز")
        status_layout = QtWidgets.QVBoxLayout()
        self.status_label = QtWidgets.QLabel("⏳ جاري التهيئة...")
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)
        
        # قائمة الطائرات المكتشفة
        self.aircraft_list_widget = AircraftListWidget()
        left_layout.addWidget(self.aircraft_list_widget)
        
        # أزرار إضافية
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶️ بدء الرادار")
        self.start_btn.clicked.connect(self.start_radar)
        self.stop_btn = QtWidgets.QPushButton("⏹️ إيقاف الرادار")
        self.stop_btn.clicked.connect(self.stop_radar)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        left_layout.addLayout(btn_layout)
        
        left_layout.addStretch()
        
        # ===== اللوحة اليمنى (عرض الطيف) =====
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # عناوين
        plot_title = QtWidgets.QLabel("<h2>📊 تحليل الطيف الترددي</h2>")
        plot_title.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(plot_title)
        
        # إعدادات PyQtGraph
        pg.setConfigOptions(antialias=True)
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground('#1a1a2e')
        self.spectrum_plot.setLabel('left', 'القوة', units='dB')
        self.spectrum_plot.setLabel('bottom', 'التردد', units='Hz')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        
        right_layout.addWidget(self.spectrum_plot)
        
        # معلومات إضافية
        info_bar = QtWidgets.QWidget()
        info_layout = QtWidgets.QHBoxLayout(info_bar)
        self.freq_info = QtWidgets.QLabel("التردد الحالي: -- MHz")
        self.power_info = QtWidgets.QLabel("أقصى قدرة: -- dB")
        info_layout.addWidget(self.freq_info)
        info_layout.addWidget(self.power_info)
        info_layout.addStretch()
        right_layout.addWidget(info_bar)
        
        # إضافة اللوحات إلى التخطيط الرئيسي
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=2)
        
        # مؤقت لتحديث الواجهة
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)
        
    def setup_sdr(self):
        """تهيئة خيط SDR"""
        self.sdr_thread = SDRThread()
        self.sdr_thread.data_received.connect(self.on_data_received)
        self.sdr_thread.error_occurred.connect(self.on_sdr_error)
        
    def apply_frequency(self):
        """تطبيق التردد المدخل"""
        try:
            freq_mhz = float(self.freq_input.text())
            self.set_frequency(freq_mhz * 1e6)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "خطأ", "الرجاء إدخال تردد صحيح (مثل: 300.0)")
            
    def set_frequency(self, freq_hz):
        """تعيين التردد المركزي"""
        self.current_freq = freq_hz
        if self.sdr_thread and self.sdr_thread.isRunning():
            self.sdr_thread.set_frequency(freq_hz)
        self.freq_info.setText(f"التردد الحالي: {freq_hz/1e6:.3f} MHz")
        
    def update_gain(self, value):
        """تحديث قيمة الكسب"""
        self.gain_label.setText(f"الكسب: {value} dB")
        if self.sdr_thread and self.sdr_thread.sdr:
            try:
                self.sdr_thread.sdr.gain = value
            except:
                pass
                
    def start_radar(self):
        """بدء تشغيل الرادار"""
        if self.sdr_thread and not self.sdr_thread.isRunning():
            self.sdr_thread.sample_rate = self.sample_rate
            self.sdr_thread.gain = self.gain_slider.value()
            self.sdr_thread.start()
            self.status_label.setText("✅ الرادار يعمل - جاري المسح")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
    def stop_radar(self):
        """إيقاف تشغيل الرادار"""
        if self.sdr_thread and self.sdr_thread.isRunning():
            self.sdr_thread.stop()
            self.status_label.setText("⏹️ الرادار متوقف")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
    def on_data_received(self, power_data, start_freq, end_freq):
        """معالجة البيانات المستلمة من SDR"""
        self.current_power = power_data
        self.current_start_freq = start_freq
        self.current_end_freq = end_freq
        
    def detect_aircraft_signals(self, power_data, freqs):
        """كشف الإشارات التي قد تكون طائرات بناءً على خصائص الإشارة"""
        if power_data is None or len(power_data) == 0:
            return []
        
        # حساب عتبة الكشف (أعلى من متوسط القدرة ب 15 dB)
        avg_power = np.mean(power_data)
        threshold = avg_power + 15
        
        peaks = []
        for i in range(1, len(power_data) - 1):
            if power_data[i] > threshold and power_data[i] > power_data[i-1] and power_data[i] > power_data[i+1]:
                freq = freqs[i]
                power = power_data[i]
                
                # تجاهل الإشارات الضعيفة جداً
                if power > threshold:
                    peaks.append((freq, power))
                    
        return peaks
        
    def identify_aircraft(self, freq):
        """محاولة تحديد نوع الطائرة بناءً على التردد (مبسط)"""
        # هذه خريطة تقريبية لترددات الطائرات العسكرية الشائعة
        freq_mhz = freq / 1e6
        
        # نطاقات تردد معروفة لأنواع الطائرات (تقريبية)
        if 225 <= freq_mhz <= 260:
            return "A-10"  # احتمالية A-10
        elif 260 <= freq_mhz <= 300:
            return "F-16"
        elif 300 <= freq_mhz <= 380:
            return "F-22"
        else:
            return "Unknown"
            
    def update_display(self):
        """تحديث الرسم البياني والكشف عن الطائرات"""
        if hasattr(self, 'current_power') and self.current_power is not None:
            # إنشاء مصفوفة الترددات
            freqs = np.linspace(self.current_start_freq, self.current_end_freq, len(self.current_power))
            
            # تحديث الرسم البياني
            if self.curve is None:
                self.curve = self.spectrum_plot.plot(freqs / 1e6, self.current_power, pen=pg.mkPen(color='#00ff00', width=2))
                self.spectrum_plot.setXRange(225, 380)  # نطاق الترددات العسكرية
                self.spectrum_plot.setYRange(0, 100)
            else:
                self.curve.setData(freqs / 1e6, self.current_power)
            
            # تحديث معلومات القدرة القصوى
            max_power = np.max(self.current_power)
            self.power_info.setText(f"أقصى قدرة: {max_power:.1f} dB")
            
            # كشف الإشارات
            peaks = self.detect_aircraft_signals(self.current_power, freqs)
            
            # إضافة علامات على القمم المكتشفة
            for peak_freq, peak_power in peaks:
                # رسم نقطة على القمة
                scatter = pg.ScatterPlotItem(
                    [peak_freq / 1e6], [peak_power],
                    size=10, brush=pg.mkBrush(255, 0, 0, 150)
                )
                self.spectrum_plot.addItem(scatter)
                
                # إضافة الطائرة إلى القائمة (إذا كانت جديدة)
                aircraft_code = self.identify_aircraft(peak_freq)
                self.aircraft_list_widget.add_aircraft(aircraft_code, peak_freq, peak_power)
                
                # تنظيف العلامات القديمة بعد فترة
                QtCore.QTimer.singleShot(3000, lambda: self.spectrum_plot.removeItem(scatter))
                
    def on_sdr_error(self, error_msg):
        """معالجة أخطاء SDR"""
        self.status_label.setText(f"⚠️ خطأ: {error_msg}")
        QtWidgets.QMessageBox.critical(self, "خطأ في SDR", error_msg)
        
    def closeEvent(self, event):
        """إغلاق النافذة وتحرير الموارد"""
        self.stop_radar()
        event.accept()
        
# ==================== تشغيل التطبيق ====================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = RadarWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
