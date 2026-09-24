# Stop the hill-climb loop, its handover wrapper and anything it launched (self-play, training, gates).
$all = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
$top = $all | Where-Object { $_.CommandLine -match 'loop\.py|selfplay\.py|train\.py|netmatch\.py|sprt_gate\.py' }
$ids = @($top | ForEach-Object { $_.ProcessId })
$kids = $all | Where-Object { $cl = $_.CommandLine; $ids | Where-Object { $cl -like "*parent_pid=$_*" } }
foreach ($p in @($top) + @($kids)) { if ($p) { try { Stop-Process -Id $p.ProcessId -Force -Confirm:$false -ErrorAction Stop; "stopped $($p.ProcessId)" } catch {} } }
