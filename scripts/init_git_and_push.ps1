# One-command: initialize Git + LFS and push the full repo (incl. neural weights)
# to GitLink. Run once, passing your repo URL.
#
#   powershell -ExecutionPolicy Bypass -File scripts/init_git_and_push.ps1 `
#       -RemoteUrl https://gitlink.org.cn/<you>/<repo>.git
#
# .gitattributes already routes *.bin/*.safetensors/*.docx/*.mp4/*.db/*.zip
# through Git LFS, so the ~3.5 GB of model weights upload as LFS objects.

param(
    [Parameter(Mandatory = $true)] [string] $RemoteUrl,
    [string] $Branch = "main"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path .git)) { git init }
git lfs install --local
git config core.autocrlf false

git add .gitattributes .gitignore
git add -A
git commit -m "MediGraph Agent: neural GPLinker extraction + full reproducible pipeline"

git branch -M $Branch
if (git remote | Select-String -Quiet '^origin$') { git remote remove origin }
git remote add origin $RemoteUrl

Write-Host "Pushing LFS objects + branch to $RemoteUrl ..."
git push -u origin $Branch

Write-Host "Done. Verify on GitLink that model weights show as LFS objects."
