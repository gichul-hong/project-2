# v10 자동 채점 감시자: 새 체크포인트가 나오면 evaluate.py로 채점해 scores.txt에 기록
$py = "C:\Users\삼성\.conda\envs\pjt-2\python.exe"
$ckptDir = "C:\hong\project-2\practice2_speed\checkpoints\bump_challenge"
$out = "C:\hong\project-2\practice2_speed\scores.txt"
$done = @{}
while ($true) {
    $files = Get-ChildItem $ckptDir -Filter "walker_model_*_steps.zip" -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime
    foreach ($f in $files) {
        if ($done.ContainsKey($f.Name)) { continue }
        # 파일이 완전히 써질 때까지 잠시 대기
        Start-Sleep -Seconds 3
        $res = & $py -u evaluate.py --model "checkpoints/bump_challenge/$($f.Name)" 2>&1
        $score = ($res | Select-String -Pattern "점수" | ForEach-Object { $_.Line }) -join " "
        $spd   = ($res | Select-String -Pattern "평균 속도" | ForEach-Object { $_.Line }) -join " "
        $steps = ($res | Select-String -Pattern "생존 스텝" | ForEach-Object { $_.Line }) -join " "
        "$($f.Name) | $score | $spd | $steps" | Add-Content -Path $out -Encoding UTF8
        $done[$f.Name] = $true
    }
    Start-Sleep -Seconds 30
}
