; Inno Setup 6 script for Eye Tracker.
;
; Per-user install into %LOCALAPPDATA%\Programs\Eye Tracker: no administrator rights,
; no UAC prompt, nothing written outside the user's profile.
;
; Build from the repository root after PyInstaller has produced dist\EyeTracker:
;
;   iscc /DAppVersion=0.1.0 packaging\windows\installer.iss
;
; Optional defines:
;   /DAppVersionNumeric=0.1.0   purely numeric version for the file properties
;                               (default: AppVersion; required for versions like 0.2.0rc1)
;   /DBundleDir=<dir>           PyInstaller output folder (default: dist\EyeTracker)
;   /DOutputDir=<dir>           where the setup program is written (default: dist)
;
; Keep this file UTF-8 *with* a byte order mark: without it Inno Setup reads the
; publisher name and the Turkish messages in the ANSI code page.

#ifndef AppVersion
  #error Pass the version, for example: iscc /DAppVersion=0.1.0 packaging\windows\installer.iss
#endif
#ifndef AppVersionNumeric
  #define AppVersionNumeric AppVersion
#endif
#ifndef BundleDir
  #define BundleDir AddBackslash(SourcePath) + "..\..\dist\EyeTracker"
#endif
#ifndef OutputDir
  #define OutputDir AddBackslash(SourcePath) + "..\..\dist"
#endif

#define AppName "Eye Tracker"
; AppId identifies the installation for upgrades and uninstall. Never change it.
#define AppGuid "F2F98F7C-0EEF-44D5-ADDE-173D9D8BA82B"
#define AppExeName "EyeTracker.exe"
#define CliExeName "eye-tracker-cli.exe"
; Same ID the app sets at run time (SetCurrentProcessExplicitAppUserModelID), so
; notifications and the taskbar group with the Start menu shortcut.
#define AppUserModelID "io.github.bugraskl.eyetracker"
#define RepoUrl "https://github.com/bugraskl/eye-tracker"
; Value name shared with eye_tracker.platform.autostart, so the app and the
; installer manage the same "start at sign-in" entry.
#define RunValueName "EyeTracker"
#define RunKey "Software\Microsoft\Windows\CurrentVersion\Run"
#define StartupApprovedKey "Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"
; Where Inno Setup records this per-user installation (used to detect upgrades).
#define UninstallKey "Software\Microsoft\Windows\CurrentVersion\Uninstall\{" + AppGuid + "}_is1"

#if !FileExists(AddBackslash(BundleDir) + AppExeName)
  #error PyInstaller output not found; build it first (see packaging/pyinstaller/eye-tracker.spec)
#endif

[Setup]
AppId={{{#AppGuid}}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Buğra Şıkel
AppPublisherURL={#RepoUrl}
AppSupportURL={#RepoUrl}/issues
AppUpdatesURL={#RepoUrl}/releases
AppCopyright=Copyright (c) 2026 Buğra Şıkel. MIT License.
AppComments=Look at a monitor and your cursor and keyboard focus follow.
VersionInfoVersion={#AppVersionNumeric}
VersionInfoProductVersion={#AppVersionNumeric}
VersionInfoProductTextVersion={#AppVersion}
VersionInfoDescription={#AppName} Setup

; Per-user only: {userpf} is %LOCALAPPDATA%\Programs.
PrivilegesRequired=lowest
DefaultDirName={userpf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
DisableReadyPage=yes
UsePreviousAppDir=yes
; Remembers the desktop-icon choice. The "startup" task is never re-applied from
; the previous installation: see ShouldWriteAutostart in [Code].
UsePreviousTasks=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763

LicenseFile=..\..\LICENSE
#if FileExists(AddBackslash(SourcePath) + "eye-tracker.ico")
SetupIconFile=eye-tracker.ico
#endif
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
WizardStyle=modern
ShowLanguageDialog=auto

OutputDir={#OutputDir}
OutputBaseFilename=EyeTracker-{#AppVersion}-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
; Anything still holding our files after "ctl quit" is released through the Restart Manager.
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[CustomMessages]
english.StartAtLogin=Start {#AppName} when I sign in to Windows
turkish.StartAtLogin=Windows'a oturum açtığımda {#AppName} uygulamasını başlat
english.AutostartGroup=Startup:
turkish.AutostartGroup=Başlangıç:
english.ShortcutComment=Glance at a monitor to move your cursor and focus there
turkish.ShortcutComment=Bir monitöre bakın; imleç ve odak oraya geçsin
english.RemoveUserData=Also delete your {#AppName} settings, calibration and logs?%n%nChoose No to keep them for a future installation.
turkish.RemoveUserData={#AppName} ayarlarınız, kalibrasyonunuz ve günlük dosyalarınız da silinsin mi?%n%nİleride yeniden kurmak üzere saklamak için Hayır'ı seçin.

[Tasks]
Name: "startup"; Description: "{cm:StartAtLogin}"; GroupDescription: "{cm:AutostartGroup}"; Flags: unchecked
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; One-folder builds change file names between releases; start from a clean
; runtime folder so no stale libraries are left behind after an upgrade.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; AppUserModelID: "{#AppUserModelID}"; Comment: "{cm:ShortcutComment}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; AppUserModelID: "{#AppUserModelID}"; Comment: "{cm:ShortcutComment}"; Tasks: desktopicon

[Registry]
; Same entry that "Start at login" in the app writes (see platform/autostart.py).
; On upgrades ShouldWriteAutostart keeps whatever the user has chosen since.
Root: HKCU; Subkey: "{#RunKey}"; ValueType: string; ValueName: "{#RunValueName}"; ValueData: """{app}\{#AppExeName}"" --background"; Flags: uninsdeletevalue; Tasks: startup; Check: ShouldWriteAutostart
; A "Disabled" flag left by Task Manager would silently override the entry above.
Root: HKCU; Subkey: "{#StartupApprovedKey}"; ValueType: none; ValueName: "{#RunValueName}"; Flags: deletevalue dontcreatekey; Tasks: startup; Check: ShouldWriteAutostart

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: dirifempty; Name: "{app}"

[Code]
const
  { platformdirs locations (appauthor "bugraskl", appname "eye-tracker"). }
  UserDataParent = '{localappdata}\bugraskl';
  UserDataDir = '{localappdata}\bugraskl\eye-tracker';
  { "eye-tracker ctl" exit code when no instance is running. }
  CtlNotRunning = 3;

var
  { An earlier version is installed for this user. }
  IsUpgrade: Boolean;
  { Start-at-login was on for this installation just before files were copied. }
  AutostartWasOn: Boolean;
  { The Select Tasks page shows the real start-at-login state (upgrades only). }
  StartupPageSynced: Boolean;

function InitializeSetup: Boolean;
begin
  IsUpgrade := RegKeyExists(HKCU, '{#UninstallKey}');
  Result := True;
end;

{ Whether the "start at sign-in" entry starts the copy in AppDir and Task Manager
  has not disabled it (in StartupApproved an odd first byte means "disabled"). }
function AutostartEnabledFor(const AppDir: String): Boolean;
var
  Command: String;
  Approved: AnsiString;
begin
  Result := False;
  if (AppDir = '') or
     not RegQueryStringValue(HKCU, '{#RunKey}', '{#RunValueName}', Command) then
    Exit;
  if Pos(Lowercase(AddBackslash(AppDir) + '{#AppExeName}'), Lowercase(Command)) = 0 then
    Exit;
  if RegQueryBinaryValue(HKCU, '{#StartupApprovedKey}', '{#RunValueName}', Approved) and
     (Length(Approved) > 0) and ((Ord(Approved[1]) and 1) = 1) then
    Exit;
  Result := True;
end;

{ What the command line says about the "startup" task: 1 selected, 0 deselected,
  -1 not mentioned. /TASKS= deselects every task it does not list; /MERGETASKS=
  changes only the tasks it names ("!startup" deselects). }
function StartupTaskParam: Integer;
var
  I, Comma: Integer;
  Param, Tasks, Task: String;
begin
  Result := -1;
  for I := 1 to ParamCount do
  begin
    Param := Lowercase(ParamStr(I));
    if Pos('/tasks=', Param) = 1 then
    begin
      Tasks := Copy(Param, Length('/tasks=') + 1, Length(Param));
      Result := 0;
    end
    else if Pos('/mergetasks=', Param) = 1 then
      Tasks := Copy(Param, Length('/mergetasks=') + 1, Length(Param))
    else
      Continue;
    Tasks := RemoveQuotes(Tasks);
    while Tasks <> '' do
    begin
      Comma := Pos(',', Tasks);
      if Comma = 0 then
        Comma := Length(Tasks) + 1;
      Task := Trim(Copy(Tasks, 1, Comma - 1));
      Delete(Tasks, 1, Comma);
      if (Task = 'startup') or (Task = '*startup') then
        Result := 1
      else if Task = '!startup' then
        Result := 0;
    end;
  end;
end;

{ Upgrades: show the real start-at-login state on the Select Tasks page instead
  of the choice made at the first installation, which UsePreviousTasks would
  restore even after the user turned autostart off in the app or Task Manager. }
procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpSelectTasks) and IsUpgrade and not StartupPageSynced then
  begin
    if AutostartEnabledFor(WizardDirValue) then
      WizardSelectTasks('startup')
    else
      WizardSelectTasks('!startup');
    StartupPageSynced := True;
  end;
end;

{ Check for the [Registry] autostart entries, evaluated when the "startup" task
  is selected. A new installation writes them as chosen. An upgrade never turns
  start-at-login back on by itself (silent upgrades such as winget re-apply the
  first installation's task selection): it writes them only when autostart is
  off now and the user asked for it, by ticking the box on a page that showed
  the real state, or with /TASKS or /MERGETASKS on a silent command line. An
  entry that is already on is left as it is (it may carry the app's options). }
function ShouldWriteAutostart: Boolean;
begin
  if not IsUpgrade then
    Result := True
  else if AutostartWasOn then
    Result := False
  else if WizardSilent or not StartupPageSynced then
    Result := StartupTaskParam = 1
  else
    Result := True;
end;

{ Ask a running Eye Tracker to quit through its local control socket and wait
  (up to about 10 s) until it has released the socket, then give it a moment to
  release the camera and its files. Does nothing if it is not running. }
procedure QuitRunningApp;
var
  Cli: String;
  ResultCode, Attempt: Integer;
begin
  Cli := ExpandConstant('{app}\{#CliExeName}');
  if not FileExists(Cli) then
    Exit;
  if (not Exec(Cli, 'ctl quit', '', SW_HIDE, ewWaitUntilTerminated, ResultCode)) or
     (ResultCode <> 0) then
    Exit;
  for Attempt := 1 to 20 do
  begin
    Sleep(500);
    if Exec(Cli, 'ctl status --timeout 500', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and
       (ResultCode = CtlNotRunning) then
      Break;
  end;
  Sleep(1000);
end;

{ Before files are replaced during an upgrade. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  QuitRunningApp;
end;

{ Remove the "start at sign-in" entries if they belong to this installation. A
  Run value pointing somewhere else (a portable copy) is left alone; an orphaned
  StartupApproved value without a Run value is removed. }
procedure RemoveAutostartIfOurs;
var
  Command: String;
begin
  if RegQueryStringValue(HKCU, '{#RunKey}', '{#RunValueName}', Command) then
  begin
    if Pos(Lowercase(ExpandConstant('{app}')), Lowercase(Command)) = 0 then
      Exit;
    RegDeleteValue(HKCU, '{#RunKey}', '{#RunValueName}');
  end;
  RegDeleteValue(HKCU, '{#StartupApprovedKey}', '{#RunValueName}');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
    { Runs before the [Registry] section, so ShouldWriteAutostart sees the old state. }
    AutostartWasOn := IsUpgrade and AutostartEnabledFor(ExpandConstant('{app}'))
  else if (CurStep = ssPostInstall) and AutostartWasOn and
          not WizardIsTaskSelected('startup') and
          ((not WizardSilent and StartupPageSynced) or (StartupTaskParam = 0)) then
    { The user unticked a box that showed start-at-login as on, or passed !startup. }
    RemoveAutostartIfOurs;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    QuitRunningApp;
    RemoveAutostartIfOurs;
  end;

  if (CurUninstallStep = usPostUninstall) and (not UninstallSilent) and
     DirExists(ExpandConstant(UserDataDir)) then
  begin
    if MsgBox(CustomMessage('RemoveUserData'), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
    begin
      DelTree(ExpandConstant(UserDataDir), True, True, True);
      RemoveDir(ExpandConstant(UserDataParent));
    end;
  end;
end;
