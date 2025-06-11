# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['front_main.py'],
    pathex=['E:\\Afeka\\FinalProject\\Project\\Investment-portfolio-management-system\\backend'],
    binaries=[],
    datas=[('logo.ico', '.'), ('logs', 'logs'), ('data', 'data'), ('..\\backend', 'backend')],
    hiddenimports=['backend.Logging_and_Validation'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SharpSight',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.ico'],
)
