# v10 언덕오르기(hill-climb) 오케스트레이터
# 아이디어: 채점이 결정적이므로 "최고 체크포인트 1개"만 중요하다.
# - 항상 현재 최고 체크포인트에서 학습을 재시작 (탐색 기점을 최고점으로 고정)
# - 새 체크포인트가 최고를 넘으면 즉시 그 지점에서 재시작 (언덕오르기)
# - 6개 연속 개선 없으면 그래도 최고점에서 재시작 (RNG가 달라져 새로운 탐색 경로 = 복권 재추첨)
$py = "C:\Users\삼성\.conda\envs\pjt-2\python.exe"
$root = "C:\hong\project-2\practice2_speed"
$ckptDir = "$root\checkpoints\bump_challenge"
$arch = "$root\checkpoints\b1_archive"
$bestDir = "C:\hong\project-2\practice2\checkpoints\v10_best"
$log = "$root\hillclimb.log"

$best = "$bestDir\walker_model_4300000_steps.zip"
$bestScore = 53.07

function Log($msg) {
    "$(Get-Date -Format 'HH:mm:ss') $msg" | Add-Content -Path $log -Encoding UTF8
}

Log "=== hill-climb 시작. best=$best ($bestScore m) ==="
$round = 0
while ($true) {
    $round++
    # 이전 산출물 정리
    Move-Item "$ckptDir\*.zip" $arch -Force -ErrorAction SilentlyContinue
    # 학습 시작 (현재 최고점에서)
    $proc = Start-Process -FilePath $py -ArgumentList @("-u", "learning.py", "--bump_challenge", "--resume", $best) `
            -WorkingDirectory $root -PassThru -WindowStyle Hidden
    Log "round $round : 학습 시작 pid=$($proc.Id) from $(Split-Path $best -Leaf)"
    $done = @{}
    $noImprove = 0
    $restart = $false
    while (-not $restart) {
        Start-Sleep -Seconds 20
        if ($proc.HasExited) { Log "round $round : 학습 프로세스 사망. 재시작"; break }
        $files = Get-ChildItem $ckptDir -Filter "walker_model_*_steps.zip" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime
        foreach ($f in $files) {
            if ($done.ContainsKey($f.Name)) { continue }
            Start-Sleep -Seconds 3
            $res = & $py -u evaluate.py --model "checkpoints/bump_challenge/$($f.Name)" 2>&1
            $line = ($res | Select-String -Pattern "점수").Line
            $score = 0.0
            if ($line -match "([\d\.]+)\s*m") { $score = [double]$matches[1] }
            Log ("round $round : $($f.Name) -> $score m" + $(if ($score -gt $bestScore) { "  ** 신기록 **" } else { "" }))
            $done[$f.Name] = $true
            if ($score -gt $bestScore) {
                $bestScore = $score
                $stamp = [int](Get-Date -UFormat %s)
                $newBest = "$bestDir\best_${score}m_$($f.Name)"
                Copy-Item $f.FullName $newBest -Force
                $best = $newBest
                Log "round $round : 새 최고 $score m -> $newBest 에서 재시작"
                $restart = $true
            } else {
                $noImprove++
                if ($noImprove -ge 6) {
                    Log "round $round : 6회 무개선 -> 최고점($bestScore m)에서 재추첨"
                    $restart = $true
                }
            }
        }
    }
    # 학습 프로세스와 워커 정리
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*multiprocessing*" -or $_.CommandLine -like "*learning.py*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 5
}
