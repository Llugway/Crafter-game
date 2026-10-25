# Launch several independent training runs in parallel, one window per run (Windows PowerShell).
# Replaces the old multiple_runs.bat (which had a hard-coded Python path).
#
#   .\scripts\run_seeds.ps1                                      # PPO, 1 env, runs 1..3, 250k steps
#   .\scripts\run_seeds.ps1 -NumEnvs 16 -Steps 1000000 -EvalInterval 50000
#                                                                # PPO, 16 parallel envs per run
#   .\scripts\run_seeds.ps1 -NumEnvs 16 -EntCoef 0.01 -Name ppo_vec16_ent001 -Steps 1000000
#   .\scripts\run_seeds.ps1 -NumEnvs 16 -Fixes 1 -Name ppo_vec16_fixes -Steps 1000000
#   .\scripts\run_seeds.ps1 -Script train_dqn.py -Agent dqn -Runs 2 -Steps 500000
#
# From cmd / Anaconda Prompt use scripts\run_seeds.bat with the same arguments.
# Run it from the repository root, with the conda env activated (conda activate crafterEnv).
#
# With -NumEnvs > 1 and no -NumWorkers, the worker processes are shared out so that all runs
# together use about (number of logical CPUs - number of runs), at least 1 per run.
param(
    [string]$Script = "train_ppo.py",
    [string]$Agent = "",               # only for train_dqn.py: dqn | ddqn
    [int]$Runs = 3,
    [int]$Steps = 250000,
    [int]$EvalInterval = 25000,
    [int]$NumEnvs = 1,                 # PPO only: environments in parallel per run
    [int]$NumWorkers = 0,              # PPO only: worker processes per run (0 = automatic)
    [double]$EntCoef = -1,             # PPO only: entropy bonus (-1 = script default)
    [int]$Fixes = 0,                   # PPO only: 1 = the three correctness fixes (--fixes 1)
    [string]$Name = "",                # results go to logdir/<Name>/<run>
    [int]$FirstRun = 1                 # number of the first run folder
)

if ($Name) { $name = $Name }
elseif ($Agent) { $name = "$($Agent)_agent" }
elseif ($NumEnvs -gt 1) { $name = "ppo_vec$NumEnvs" }
else { $name = "ppo_agent" }

if ($NumEnvs -gt 1 -and $NumWorkers -le 0) {
    $cpus = [Environment]::ProcessorCount
    $NumWorkers = [Math]::Max(1, [Math]::Min($NumEnvs, [Math]::Floor(($cpus - $Runs) / $Runs)))
}

for ($i = $FirstRun; $i -lt $FirstRun + $Runs; $i++) {
    $cmdArgs = @($Script, "--steps", $Steps, "--eval-interval", $EvalInterval, "--logdir", "logdir/$name/$i")
    if ($Agent) { $cmdArgs += @("--agent", $Agent) }
    if ($NumEnvs -gt 1) { $cmdArgs += @("--num-envs", $NumEnvs, "--num-workers", $NumWorkers) }
    # invariant culture: on a French Windows, 0.01 would otherwise be written "0,01"
    if ($Fixes -gt 0) { $cmdArgs += @("--fixes", 1) }
    if ($EntCoef -ge 0) { $cmdArgs += @("--ent-coef", $EntCoef.ToString([Globalization.CultureInfo]::InvariantCulture)) }
    Start-Process -FilePath "python" -ArgumentList $cmdArgs
    Write-Host "started run $i -> logdir/$name/$i   (python $($cmdArgs -join ' '))"
}
