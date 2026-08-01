param(
    [int[]]$Seeds = @(45, 46, 47),
    [int]$Epochs = 60,
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = '.\八、运行环境与配置说明\运行环境\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

$Data = '.\九、处理后磁触觉数据集'
$Batch = 32
$SeqLen = 1024
$Lr = '1e-3'
$PhyLoss = 'mae'
$NumWorkers = 0
$UsePinMemory = $true

$TrainSuffixes = @('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12')
$ValSuffixes = @('13', '14')
$TestSuffixes = @('15', '16', '17', '18', '19', '20')

$OutDir = ".\十、实验结果\多尺度门控交互网络消融实验_ep${Epochs}_$($Seeds.Count)seeds"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Conditions = @(
    @{
        Group = 'benchmark'
        Name = 'full_msgi'
        Model = 'msgi_net'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'All modules'
        Change = 'Full model: magnetic16 input, conv stem, 4 MSGIBlocks, multiscale gated interaction, classification/regression heads'
        Role = 'Baseline for all ablation variants'
        ConclusionHint = 'Full-model baseline'
    },
    @{
        Group = 'task'
        Name = 'cls_only'
        Model = 'msgi_net'
        TaskMode = 'cls_only'
        PhyWeight = '0.0'
        ChannelPreset = 'magnetic16'
        Module = 'Force regression head and regression loss'
        Change = 'Remove auxiliary force-regression head and regression loss; keep classification supervision only'
        Role = 'Test whether continuous force supervision improves physical consistency of shared features'
        ConclusionHint = 'If classification drops, multitask physical supervision is useful'
    },
    @{
        Group = 'input'
        Name = 'total4'
        Model = 'msgi_net'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'total4'
        Module = 'XYZ components'
        Change = 'Keep only four total-field channels and remove XYZ components'
        Role = 'Test contribution of direction-aware magnetic components'
        ConclusionHint = 'A drop indicates XYZ components are necessary'
    },
    @{
        Group = 'input'
        Name = 'xyz12'
        Model = 'msgi_net'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'xyz12'
        Module = 'Total-field channels'
        Change = 'Keep only XYZ components and remove total-field channels'
        Role = 'Test contribution of total-field magnitude information'
        ConclusionHint = 'A drop indicates total-field information complements XYZ components'
    },
    @{
        Group = 'multiscale'
        Name = 'single_scale'
        Model = 'msgi_single_scale'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'Parallel k=3/k=7 multiscale structure'
        Change = 'Replace parallel multiscale structure with a single-scale convolution'
        Role = 'Test whether multiscale receptive fields outperform single-scale modeling'
        ConclusionHint = 'A drop indicates the multiscale structure is useful'
    },
    @{
        Group = 'multiscale'
        Name = 'k3_only'
        Model = 'msgi_k3_only'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'k=7 large-scale branch'
        Change = 'Keep only k=3 branch and remove k=7 large-scale branch'
        Role = 'Test contribution of large receptive field to long-context loading dynamics'
        ConclusionHint = 'A drop indicates the large-scale context branch is important'
    },
    @{
        Group = 'multiscale'
        Name = 'k7_only'
        Model = 'msgi_k7_only'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'k=3 small-scale branch'
        Change = 'Keep only k=7 branch and remove k=3 small-scale branch'
        Role = 'Test contribution of small receptive field to local transient details'
        ConclusionHint = 'Used to judge the gain boundary of the small-scale branch'
    },
    @{
        Group = 'gating'
        Name = 'no_gate'
        Model = 'msgi_no_gate'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'Sigmoid gated interaction'
        Change = 'Keep multiscale convolutions and remove u times sigmoid(gate) modulation'
        Role = 'Test adaptive selection of key channels and time positions by gating'
        ConclusionHint = 'A drop indicates gated interaction is useful'
    },
    @{
        Group = 'depth'
        Name = 'layers_2'
        Model = 'msgi_2layers'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = '4x MSGIBlock trunk depth'
        Change = 'Reduce trunk depth from 4 MSGIBlocks to 2'
        Role = 'Test whether a shallow trunk is under-expressive'
        ConclusionHint = 'A drop indicates deeper temporal abstraction is necessary'
    },
    @{
        Group = 'depth'
        Name = 'layers_6'
        Model = 'msgi_6layers'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = '4x MSGIBlock trunk depth'
        Change = 'Increase trunk depth from 4 MSGIBlocks to 6'
        Role = 'Test whether a deeper trunk brings further gains'
        ConclusionHint = 'No gain indicates 4 blocks balance capacity and stability'
    },
    @{
        Group = 'phy_weight'
        Name = 'phy0p00'
        Model = 'msgi_net'
        TaskMode = 'multitask'
        PhyWeight = '0.0'
        ChannelPreset = 'magnetic16'
        Module = 'Physical supervision weight lambda'
        Change = 'Set regression-loss weight to 0.0 while keeping the regression head'
        Role = 'Test multitask architecture without physical supervision'
        ConclusionHint = 'Lower bound for lambda ablation'
    },
    @{
        Group = 'phy_weight'
        Name = 'phy0p05'
        Model = 'msgi_net'
        TaskMode = 'multitask'
        PhyWeight = '0.05'
        ChannelPreset = 'magnetic16'
        Module = 'Physical supervision weight lambda'
        Change = 'Set regression-loss weight to 0.05'
        Role = 'Test whether weak physical supervision is sufficient'
        ConclusionHint = 'Used to show whether too-small lambda is insufficient'
    },
    @{
        Group = 'phy_weight'
        Name = 'phy0p20'
        Model = 'msgi_net'
        TaskMode = 'multitask'
        PhyWeight = '0.2'
        ChannelPreset = 'magnetic16'
        Module = 'Physical supervision weight lambda'
        Change = 'Set regression-loss weight to 0.2'
        Role = 'Test whether stronger physical supervision affects classification'
        ConclusionHint = 'Used to show classification/regression tradeoff as lambda grows'
    },
    @{
        Group = 'phy_weight'
        Name = 'phy0p50'
        Model = 'msgi_net'
        TaskMode = 'multitask'
        PhyWeight = '0.5'
        ChannelPreset = 'magnetic16'
        Module = 'Physical supervision weight lambda'
        Change = 'Set regression-loss weight to 0.5'
        Role = 'Test whether overly strong physical supervision suppresses classification'
        ConclusionHint = 'Used to show lambda should not be too large'
    },
    @{
        Group = 'stability'
        Name = 'no_residual'
        Model = 'msgi_no_residual'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'Residual connection'
        Change = 'Remove residual addition in MSGIBlock while keeping BN'
        Role = 'Test contribution of residual connection to feature propagation and stability'
        ConclusionHint = 'A drop indicates residual connection is necessary'
    },
    @{
        Group = 'stability'
        Name = 'no_bn'
        Model = 'msgi_no_bn'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'BatchNorm'
        Change = 'Remove output BatchNorm in MSGIBlock while keeping residual connection'
        Role = 'Test stabilization from normalization under cross-batch magnetic variation'
        ConclusionHint = 'A drop indicates BN helps stabilize training'
    },
    @{
        Group = 'stability'
        Name = 'no_resnorm'
        Model = 'msgi_no_resnorm'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'Residual connection + BatchNorm'
        Change = 'Remove both residual connection and BatchNorm in MSGIBlock'
        Role = 'Test overall necessity of stabilization structure'
        ConclusionHint = 'A clear drop indicates residual+normalization is important'
    },
    @{
        Group = 'frontend'
        Name = 'no_conv_stem'
        Model = 'msgi_no_conv_stem'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'Conv1d(k=7)+BN+SiLU stem'
        Change = 'Replace k=7 convolutional stem with 1x1 pointwise projection'
        Role = 'Test contribution of conv stem to initial local temporal extraction'
        ConclusionHint = 'A drop indicates local convolution in the stem is useful'
    },
    @{
        Group = 'pooling'
        Name = 'max_pool'
        Model = 'msgi_max_pool'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'AdaptiveAvgPool1d global aggregation'
        Change = 'Replace global average pooling with global max pooling'
        Role = 'Test difference between average aggregation and peak aggregation'
        ConclusionHint = 'Used to test whether average pooling is more robust'
    },
    @{
        Group = 'pooling'
        Name = 'last_pool'
        Model = 'msgi_last_pool'
        TaskMode = 'multitask'
        PhyWeight = '0.1'
        ChannelPreset = 'magnetic16'
        Module = 'AdaptiveAvgPool1d global aggregation'
        Change = 'Use the last time-position feature instead of global average pooling'
        Role = 'Test whether whole-sequence global aggregation is needed'
        ConclusionHint = 'A drop indicates global temporal aggregation is necessary'
    }
)

Write-Host '========================================'
Write-Host 'Running full MSGI-Net ablation suite'
Write-Host "ProjectRoot: $ProjectRoot"
Write-Host "Python: $Python"
Write-Host "Split: train=01-12 | val=13-14 | test=15-20"
Write-Host "Seeds: $($Seeds -join ', ')"
Write-Host "Epochs: $Epochs"
Write-Host "Conditions: $($Conditions.Name -join ', ')"
Write-Host "Output: $OutDir"
Write-Host '========================================'

foreach ($Condition in $Conditions) {
    $ConditionDir = Join-Path $OutDir $Condition.Group
    New-Item -ItemType Directory -Force -Path $ConditionDir | Out-Null

    foreach ($Seed in $Seeds) {
        $OutCsv = Join-Path $ConditionDir "$($Condition.Name)_seed${Seed}.csv"
        if ((Test-Path $OutCsv) -and (-not $Force)) {
            Write-Host "`n[skip] $($Condition.Group)/$($Condition.Name) seed=$Seed already exists: $OutCsv"
            continue
        }

        $Args = @(
            '.\六、模型训练与固定跨组评估程序\compare_model_zoo_fixed_suffix_split.py',
            '--data', $Data,
            '--channel_preset', $Condition.ChannelPreset,
            '--models', $Condition.Model,
            '--epochs', $Epochs,
            '--batch', $Batch,
            '--seq_len', $SeqLen,
            '--lr', $Lr,
            '--task_mode', $Condition.TaskMode,
            '--phy_weight', $Condition.PhyWeight,
            '--phy_loss', $PhyLoss,
            '--seed', $Seed,
            '--device', $Device,
            '--num_workers', $NumWorkers,
            '--save_detailed_eval',
            '--eval_split', 'test',
            '--out_csv', $OutCsv,
            '--train_suffixes'
        ) + $TrainSuffixes + @(
            '--val_suffixes'
        ) + $ValSuffixes + @(
            '--test_suffixes'
        ) + $TestSuffixes

        if ($UsePinMemory) {
            $Args += '--pin_memory'
        }

        Write-Host "`n==================== [$($Condition.Group)] $($Condition.Name) | Seed $Seed ===================="
        Write-Host "$Python $($Args -join ' ')"

        if (-not $DryRun) {
            & $Python @Args
            if ($LASTEXITCODE -ne 0) {
                throw "Run failed for group=$($Condition.Group), condition=$($Condition.Name), seed=$Seed"
            }
            if (-not (Test-Path $OutCsv)) {
                throw "Expected result CSV was not created: $OutCsv"
            }
        }
    }
}

if ($DryRun) {
    Write-Host "`nDryRun finished. No training was executed."
    return
}

$RunsCsv = Join-Path $OutDir 'all_ablation_runs.csv'
$StatsCsv = Join-Path $OutDir 'all_ablation_stats.csv'
$TableCsv = Join-Path $OutDir 'all_ablation_total_table.csv'
$TableMd = Join-Path $OutDir 'all_ablation_total_table.md'
$TableTex = Join-Path $OutDir 'all_ablation_total_table.tex'
$FigureDir = Join-Path $OutDir 'summary_figures'
$TempPy = Join-Path $OutDir '_tmp_build_all_ablation_summary.py'
$ConditionMetaLiteral = $Conditions | ConvertTo-Json -Depth 4
$SeedListLiteral = (($Seeds | ForEach-Object { [string]$_ }) -join ', ')

$PyCode = @"
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

project_root = Path(r"$ProjectRoot")
preprocess_dir = project_root / "四、磁场信号预处理程序"
evaluation_dir = project_root / "六、模型训练与固定跨组评估程序"
for search_path in (preprocess_dir, evaluation_dir):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from preprocess import SequenceDataset
from compare_model_zoo import build_model

out_dir = Path(r"$OutDir")
conditions = json.loads(r'''$ConditionMetaLiteral''')
seeds = [$SeedListLiteral]

def read_one_row(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[0]

def read_optional_metric(path, key):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    value = rows[0].get(key, "")
    return float(value) if value not in ("", None) else None

def safe_float(value):
    return float(value) if value not in ("", None) else None

def avg(values):
    items = [v for v in values if v is not None]
    return mean(items) if items else None

def std(values):
    items = [v for v in values if v is not None]
    return stdev(items) if len(items) > 1 else 0.0 if len(items) == 1 else None

def fmt_pct(mean_value, std_value):
    if mean_value is None:
        return "--"
    return f"{mean_value * 100:.2f} ? {std_value * 100:.2f}"

def fmt_num(mean_value, std_value, digits=2):
    if mean_value is None:
        return "--"
    return f"{mean_value:.{digits}f} ? {std_value:.{digits}f}"

def fmt_delta(value):
    if value is None:
        return "--"
    return f"{value:+.2f}"

param_cache = {}
def param_count(cond):
    key = (cond["Model"], cond["TaskMode"], cond["ChannelPreset"])
    if key in param_cache:
        return param_cache[key]
    ds = SequenceDataset(r"$Data", seq_len=$SeqLen, channel_preset=cond["ChannelPreset"])
    model = build_model(
        name=cond["Model"],
        num_classes=len(ds.class_to_idx),
        seq_len=$SeqLen,
        in_channels=getattr(ds, "num_channels", 1),
        task_mode=cond["TaskMode"],
    )
    count = sum(p.numel() for p in model.parameters())
    param_cache[key] = count
    return count

run_rows = []
for cond in conditions:
    for seed in seeds:
        csv_path = out_dir / cond["Group"] / f"{cond['Name']}_seed{seed}.csv"
        if not csv_path.exists():
            raise RuntimeError(f"Missing result CSV: {csv_path}")
        row = read_one_row(csv_path)
        detail_dir = csv_path.parent / f"{csv_path.stem}_details" / cond["Model"] / "test"
        summary_path = detail_dir / "summary_metrics.csv"
        force_path = detail_dir / "force_regression_summary.csv"
        run_rows.append({
            "group": cond["Group"],
            "condition": cond["Name"],
            "model_name": cond["Model"],
            "task_mode": cond["TaskMode"],
            "phy_weight": cond["PhyWeight"],
            "channel_preset": cond["ChannelPreset"],
            "module": cond["Module"],
            "change": cond["Change"],
            "role": cond["Role"],
            "conclusion_hint": cond["ConclusionHint"],
            "seed": seed,
            "best_epoch": row.get("best_epoch", ""),
            "best_val_loss": row.get("best_val_loss", ""),
            "best_val_acc": row.get("best_val_acc", ""),
            "test_loss": row.get("test_loss", ""),
            "test_acc": row.get("test_acc", ""),
            "test_ce": row.get("test_ce", ""),
            "test_phy": row.get("test_phy", ""),
            "macro_f1": read_optional_metric(summary_path, "macro_f1"),
            "weighted_f1": read_optional_metric(summary_path, "weighted_f1"),
            "force_mae": read_optional_metric(force_path, "mae"),
            "force_r2": read_optional_metric(force_path, "r2"),
        })

with Path(r"$RunsCsv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
    writer.writeheader()
    writer.writerows(run_rows)

stat_rows = []
for cond in conditions:
    rows = [r for r in run_rows if r["group"] == cond["Group"] and r["condition"] == cond["Name"]]
    stat_rows.append({
        "group": cond["Group"],
        "condition": cond["Name"],
        "model_name": cond["Model"],
        "task_mode": cond["TaskMode"],
        "phy_weight": cond["PhyWeight"],
        "channel_preset": cond["ChannelPreset"],
        "module": cond["Module"],
        "change": cond["Change"],
        "role": cond["Role"],
        "conclusion_hint": cond["ConclusionHint"],
        "params": param_count(cond),
        "n_runs": len(rows),
        "best_epoch_mean": avg([safe_float(r["best_epoch"]) for r in rows]),
        "best_epoch_std": std([safe_float(r["best_epoch"]) for r in rows]),
        "best_val_loss_mean": avg([safe_float(r["best_val_loss"]) for r in rows]),
        "best_val_loss_std": std([safe_float(r["best_val_loss"]) for r in rows]),
        "best_val_acc_mean": avg([safe_float(r["best_val_acc"]) for r in rows]),
        "best_val_acc_std": std([safe_float(r["best_val_acc"]) for r in rows]),
        "test_acc_mean": avg([safe_float(r["test_acc"]) for r in rows]),
        "test_acc_std": std([safe_float(r["test_acc"]) for r in rows]),
        "macro_f1_mean": avg([r["macro_f1"] for r in rows]),
        "macro_f1_std": std([r["macro_f1"] for r in rows]),
        "force_mae_mean": avg([r["force_mae"] for r in rows]),
        "force_mae_std": std([r["force_mae"] for r in rows]),
        "force_r2_mean": avg([r["force_r2"] for r in rows]),
        "force_r2_std": std([r["force_r2"] for r in rows]),
    })

with Path(r"$StatsCsv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
    writer.writeheader()
    writer.writerows(stat_rows)

baseline = next(r for r in stat_rows if r["group"] == "benchmark" and r["condition"] == "full_msgi")
base_acc = baseline["test_acc_mean"]
base_f1 = baseline["macro_f1_mean"]

table_rows = []
for r in stat_rows:
    delta_acc = (r["test_acc_mean"] - base_acc) * 100 if r["test_acc_mean"] is not None and base_acc is not None else None
    delta_f1 = (r["macro_f1_mean"] - base_f1) * 100 if r["macro_f1_mean"] is not None and base_f1 is not None else None
    table_rows.append({
        "group": r["group"],
        "variant": r["condition"],
        "module": r["module"],
        "change": r["change"],
        "role": r["role"],
        "params": r["params"],
        "Test Acc (%)": fmt_pct(r["test_acc_mean"], r["test_acc_std"]),
        "Macro-F1 (%)": fmt_pct(r["macro_f1_mean"], r["macro_f1_std"]),
        "Force MAE (N)": fmt_num(r["force_mae_mean"], r["force_mae_std"]),
        "Force R2": fmt_num(r["force_r2_mean"], r["force_r2_std"]),
        "Delta Acc": fmt_delta(delta_acc),
        "Delta F1": fmt_delta(delta_f1),
        "conclusion_hint": r["conclusion_hint"],
    })

with Path(r"$TableCsv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
    writer.writeheader()
    writer.writerows(table_rows)

md_headers = ["??", "??", "????", "??/????", "????", "???", "Test Acc (%)", "Macro-F1 (%)", "Force MAE (N)", "Force R2", "?Acc", "?F1", "????"]
md_keys = ["group", "variant", "module", "change", "role", "params", "Test Acc (%)", "Macro-F1 (%)", "Force MAE (N)", "Force R2", "Delta Acc", "Delta F1", "conclusion_hint"]
lines = []
lines.append("| " + " | ".join(md_headers) + " |")
lines.append("|" + "|".join(["---"] * len(md_headers)) + "|")
for row in table_rows:
    lines.append("| " + " | ".join(str(row[k]).replace("|", "/") for k in md_keys) + " |")
Path(r"$TableMd").write_text("\n".join(lines) + "\n", encoding="utf-8")

tex_lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{MSGI-Net???????????}",
    r"\label{tab:msginet_all_ablation}",
    r"\scriptsize",
    r"\begin{tabular}{llllrrrr}",
    r"\toprule",
    r"?? & ?? & ???? & ??? & Test Acc(\%) & Macro-F1(\%) & Force MAE(N) & $\Delta$Acc \\",
    r"\midrule",
]
for row in table_rows:
    variant_tex = row['variant'].replace('_', r'\_')
    module_tex = row['module'].replace('_', r'\_')
    tex_lines.append(
        f"{row['group']} & {variant_tex} & "
        f"{module_tex} & {row['params']} & "
        f"{row['Test Acc (%)']} & {row['Macro-F1 (%)']} & {row['Force MAE (N)']} & {row['Delta Acc']} \\\\"
    )
tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
Path(r"$TableTex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

print("Wrote " + r"$RunsCsv")
print("Wrote " + r"$StatsCsv")
print("Wrote " + r"$TableCsv")
print("Wrote " + r"$TableMd")
print("Wrote " + r"$TableTex")
"@

Set-Content -Path $TempPy -Value $PyCode -Encoding UTF8
& $Python $TempPy
if ($LASTEXITCODE -ne 0) {
    throw "Failed to aggregate all ablation results"
}
Remove-Item -LiteralPath $TempPy -Force

$PlotArgs = @(
    '.\六、模型训练与固定跨组评估程序\plot_msginet_all_ablation_summary.py',
    '--results_dir', $OutDir,
    '--out_dir', $FigureDir,
    '--seeds'
) + ($Seeds | ForEach-Object { [string]$_ })

Write-Host "`nGenerating ablation summary figures ..."
Write-Host "$Python $($PlotArgs -join ' ')"
& $Python @PlotArgs
if ($LASTEXITCODE -ne 0) {
    throw "Failed to generate ablation summary figures"
}

Write-Host "`nAll ablation runs finished."
Write-Host "Runs: $RunsCsv"
Write-Host "Stats: $StatsCsv"
Write-Host "Total table CSV: $TableCsv"
Write-Host "Total table Markdown: $TableMd"
Write-Host "Total table LaTeX: $TableTex"
Write-Host "Summary figures: $FigureDir"

