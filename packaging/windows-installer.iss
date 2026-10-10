; Inno Setup script for the Windows installer (Aria-Windows-Setup.exe).
; Built by .github/workflows/apps.yml after PyInstaller has made dist\Aria:
;
;     iscc /DAppVersion=0.1.0 packaging\windows-installer.iss
;
; It installs for the current user only (no administrator password needed),
; adds Aria to the Start menu, and can put her on the desktop. Uninstalling
; removes the program but not her memory in %APPDATA%\Aria.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{5B0C8E1D-7A41-4C55-9C2E-AF1E3B6D2A90}
AppName=Aria
AppVersion={#AppVersion}
AppVerName=Aria {#AppVersion}
AppPublisher=Aria
DefaultDirName={localappdata}\Programs\Aria
DefaultGroupName=Aria
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=Aria-Windows-Setup
SetupIconFile=..\dist\aria.ico
UninstallDisplayIcon={app}\Aria.exe
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Put Aria on the desktop"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\Aria\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; An update replaces the old program files completely.
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\Aria"; Filename: "{app}\Aria.exe"
Name: "{autodesktop}\Aria"; Filename: "{app}\Aria.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Aria.exe"; Description: "Open Aria now"; Flags: nowait postinstall skipifsilent
