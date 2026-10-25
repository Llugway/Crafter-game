# Launch several independent training runs in parallel (Windows PowerShell).
# Replaces the old multiple_runs.bat (which had a hard-coded Python path).
#
#   .\scripts\run_seeds.ps1                                # PPO, runs 1..3, 250k steps
#   .\scripts\run_seeds.ps1 -Script train_dqn.py -Agent dqn -Runs 2 -Steps 500000
#
# Run it from the repository root, with the conda env activated (conda activate crafterEnv).
param(
    [string]$Script = "train_ppo.py",
    [string]$Agent = "",               # only for train_dqn.py: dqn | ddqn
    [int]$Runs = 3,
    [int]$Steps = 250000,
    [int]$EvalInterval = 25000
)
$name = if ($Agent) { "$($Agent)_agent" } else { "ppo_agent" }
for ($i = 1; $i -le $Runs; $i++) {
    $cmdArgs = @($Script, "--steps", $Steps, "--eval-interval", $EvalInterval, "--logdir", "logdir/$name/$i")
    if ($Agent) { $cmdArgs += @("--agent", $Agent) }
    Start-Process -FilePath "python" -ArgumentList $cmdArgs
    Write-Host "started run $i -> logdir/$name/$i"
}
