$cur = [Environment]::GetEnvironmentVariable('Path','User')
if (-not ($cur -like '*PortableGit*')) {
  [Environment]::SetEnvironmentVariable('Path', $cur + ';C:\Users\hjaca\PortableGit\cmd', 'User')
  Write-Output PATH_UPDATED
} else {
  Write-Output already_there
}
