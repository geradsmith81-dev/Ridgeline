# Creates "Ridgeline.lnk" on the Desktop and in the Start menu.
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ws = New-Object -ComObject WScript.Shell
$targets = @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))
# a OneDrive-redirected Desktop is what the screen shows; add it too if it is a different folder
$od = Join-Path $env:USERPROFILE 'OneDrive\Desktop'
if ((Test-Path $od) -and ($targets -notcontains $od)) { $targets += $od }
foreach ($dir in $targets) {
    $s = $ws.CreateShortcut((Join-Path $dir 'Ridgeline.lnk'))
    $s.TargetPath = $env:ComSpec
    $s.Arguments = '/c "' + (Join-Path $here 'Start app.bat') + '"'
    $s.WorkingDirectory = $here
    $s.IconLocation = (Join-Path $here 'assets\ridgeline_icon.ico')
    $s.WindowStyle = 7
    $s.Save()
}
