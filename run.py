import os
import sys

#으후루꾸꾸루후으으후루꾸꾸루후으으후루꾸꾸루후으으후루꾸꾸루후으으후루꾸꾸루후으으후루꾸꾸루후으으후루꾸꾸루후으
#2025.06.09
#으후루꾸꾸루후으으후루꾸꾸루후으으후루꾸꾸루후으으후루꾸꾸루후으으후루꾸꾸루후으으후루꾸꾸루후으으후루꾸꾸루후으


BOOL_DEBUG = True #False
# 특수한 환경이 아닌 경우엔 False로 설정하고 사용하십시오.

if BOOL_DEBUG: ### 디버그 모드 : 경로 설치가 잘 안되는 경우에 이용
    # PyQt5 설치 경로를 Python 경로에 추가
    PYQT_PATH = r'C:\Python310\Lib\site-packages'
    if not os.path.exists(PYQT_PATH):
        print(f"경고: PyQt5 설치 경로를 찾을 수 없습니다: {PYQT_PATH}")
        sys.exit(1)

    # Qt WebEngine 설정
    qt_path = os.path.join(PYQT_PATH, 'PyQt5', 'Qt5')

    # 환경 변수 설정
    os.environ['QTWEBENGINEPROCESS_PATH'] = os.path.join(qt_path, 'bin', 'QtWebEngineProcess.exe')
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--single-process'
    os.environ['QTWEBENGINE_RESOURCES_PATH'] = os.path.join(qt_path, 'resources')
    os.environ['QTWEBENGINE_LOCALES_PATH'] = os.path.join(qt_path, 'translations', 'qtwebengine_locales')
    os.environ['QTWEBENGINE_ICU_DATA_PATH'] = os.path.join(qt_path, 'bin')

    print(f"PyQt5 경로: {PYQT_PATH}")
    print(f"QtWebEngineProcess.exe 경로: {os.environ['QTWEBENGINEPROCESS_PATH']}")
    print(f"리소스 경로: {os.environ['QTWEBENGINE_RESOURCES_PATH']}")
    print(f"로케일 경로: {os.environ['QTWEBENGINE_LOCALES_PATH']}")

    sys.path.insert(0, PYQT_PATH)

import PyQt5
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineProfile
from code.Front.index import FireGuardApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = FireGuardApp()
    mainWin.show()
    sys.exit(app.exec_()) 