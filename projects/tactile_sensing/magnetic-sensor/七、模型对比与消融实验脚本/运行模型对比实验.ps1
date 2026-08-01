param(
    [int]$Epochs = 60,
    [int]$Seed = 45,
    [int]$Runs = 3,
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = '.\八、运行环境与配置说明\运行环境\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

$Data = '.\九、处理后磁触觉数据集'
if (-not (Test-Path $Data)) {
    throw "Data directory not found: $Data"
}

$Models = @(
    'baseline_1dcnn',
    'tcn',
    'mamba_like',
    'msgi_net',
    'lstm',
    'gru',
    'bilstm',
    'knn',
    'random_forest',
    'linear_svm'
)

$ChannelPreset = 'magnetic16'
$Batch = 32
$SeqLen = 1024
$Lr = '1e-3'
$TaskMode = 'multitask'
$PhyWeight = '0.1'
$PhyLoss = 'mae'
$NumWorkers = 0
$UsePinMemory = $true
$UsePersistentWorkers = $false
$UseNoDeterministic = $true

$TrainSuffixes = @('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12')
$ValSuffixes = @('13', '14')
$TestSuffixes = @('15', '16', '17', '18', '19', '20')

$ResultsRoot = ".\十、实验结果\十种模型固定跨组对比_ep${Epochs}_seed${Seed}x${Runs}"
$RunLogsDir = Join-Path $ResultsRoot 'run_logs'
$SummaryCsv = Join-Path $ResultsRoot 'run_id_mapping.csv'

function Join-Items {
    param([object[]]$Items)
    return ($Items | ForEach-Object { [string]$_ }) -join ', '
}

function Invoke-CommandOrDryRun {
    param(
        [string]$Exe,
        [object[]]$CommandArgs,
        [string]$LogPath = ''
    )

    $CommandText = "$Exe $($CommandArgs -join ' ')"
    Write-Host $CommandText

    if ($DryRun) {
        return
    }

    if ($LogPath) {
        & $Exe @CommandArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    } else {
        & $Exe @CommandArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $RunLogsDir | Out-Null

$RunIds = @()
for ($Index = 1; $Index -le $Runs; $Index++) {
    $RunIds += [int]("$Seed$Index")
}

Write-Host '========================================'
Write-Host 'Running 10-model fixed-split comparison'
Write-Host "ProjectRoot: $ProjectRoot"
Write-Host "Python: $Python"
Write-Host "Data: $Data"
Write-Host "Models: $(Join-Items $Models)"
Write-Host "Actual training seed: $Seed"
Write-Host "Run count: $Runs"
Write-Host "Virtual run ids: $(Join-Items $RunIds)"
Write-Host "Epochs: $Epochs"
Write-Host "Split: train=01-12 | val=13-14 | test=15-20"
Write-Host "No deterministic: $UseNoDeterministic"
Write-Host "ResultsRoot: $ResultsRoot"
Write-Host "DryRun: $DryRun"
Write-Host '========================================'

$MappingRows = @()

for ($Index = 0; $Index -lt $RunIds.Count; $Index++) {
    $RunId = $RunIds[$Index]
    $OutCsv = Join-Path $ResultsRoot "model_compare_seed${RunId}.csv"
    $LogPath = Join-Path $RunLogsDir "run_${RunId}.log"

    $Args = @(
        '.\六、模型训练与固定跨组评估程序\compare_model_zoo_fixed_suffix_split.py',
        '--data', $Data,
        '--channel_preset', $ChannelPreset,
        '--models'
    ) + $Models + @(
        '--epochs', $Epochs,
        '--batch', $Batch,
        '--seq_len', $SeqLen,
        '--lr', $Lr,
        '--task_mode', $TaskMode,
        '--phy_weight', $PhyWeight,
        '--phy_loss', $PhyLoss,
        '--seed', $Seed,
        '--device', $Device,
        '--num_workers', $NumWorkers,
        '--save_detailed_eval',
        '--eval_split', 'test',
        '--train_suffixes'
    ) + $TrainSuffixes + @(
        '--val_suffixes'
    ) + $ValSuffixes + @(
        '--test_suffixes'
    ) + $TestSuffixes + @(
        '--out_csv', $OutCsv
    )

    if ($UsePinMemory) {
        $Args += '--pin_memory'
    }
    if ($UsePersistentWorkers) {
        $Args += '--persistent_workers'
    }
    if ($UseNoDeterministic) {
        $Args += '--no-deterministic'
    }

    Write-Host "`n==================== Run $($Index + 1)/$Runs | RunId $RunId ===================="
    Invoke-CommandOrDryRun -Exe $Python -CommandArgs $Args -LogPath $LogPath

    $MappingRows += [PSCustomObject]@{
        run_index = $Index + 1
        virtual_seed_id = $RunId
        actual_seed = $Seed
        deterministic = (-not $UseNoDeterministic)
        out_csv = $OutCsv
        log_path = $LogPath
    }
}

$AggArgs = @(
    '.\六、模型训练与固定跨组评估程序\aggregate_model_compare_results.py',
    '--out_dir', $ResultsRoot,
    '--models'
) + $Models + @(
    '--seeds'
) + ($RunIds | ForEach-Object { [string]$_ }) + @(
    '--eval_split', 'test'
)

Write-Host "`nAggregating results ..."
Invoke-CommandOrDryRun -Exe $Python -CommandArgs $AggArgs -LogPath (Join-Path $RunLogsDir 'aggregate.log')

if (-not $DryRun) {
    $MappingRows | Export-Csv -Path $SummaryCsv -NoTypeInformation -Encoding UTF8
    Write-Host "Run-id mapping saved: $SummaryCsv"
    Write-Host "Aggregated mean: $(Join-Path $ResultsRoot 'aggregated_mean.csv')"
    Write-Host "Aggregated stats: $(Join-Path $ResultsRoot 'aggregated_stats.csv')"
    Write-Host "Aggregated runs: $(Join-Path $ResultsRoot 'aggregated_runs.csv')"
} else {
    Write-Host "`nDryRun finished. No training was executed."
}
