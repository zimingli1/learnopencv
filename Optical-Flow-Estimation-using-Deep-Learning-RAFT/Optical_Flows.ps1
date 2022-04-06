# Windows Powershell
# Ziming Li
param($inputFolderDir)

# activate python virtual environment
venv\Scripts\activate.ps1

# iterate each raw car crash videos and generate optical flow videos in the current path
# for ($i=0; $i -lt $files.Count; $i++) {
#     Write-Host $files[$i].name
# }

# iterate each raw car crash videos and generate optical flow videos in the current path
Foreach($file in Get-ChildItem $inputFolderDir) {
    Write-Host "Start in Powershell: " + $file.name
    $params = @(
        'inference.py'
        '--model'
        'models\pretrained_models\raft-kitti.pth'
        '--video'
        $inputFolderDir+'\'+$file.name
        '--save')
    Write-Host "python" $params
    python @params
    Write-Host "Done in Powershell" + $file.name
}

# deactivate python virtual environment
deactivate

Write-Output "Successfully Run the Powershell Scirpt, Check Your Output Folder"