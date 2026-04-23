param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$SourceRoot = ".\data\extracted\mvtec\mvtec",
    [string]$TargetRoot = ".\data\fewshot\mvtec_1shot",
    [int]$NumSamples = 1,
    [int]$Seed = 42
)

& $Python "foundad/src/sample.py" `
    source=$SourceRoot `
    target=$TargetRoot `
    num_samples=$NumSamples `
    seed=$Seed
