$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot '..\serial-recorder.ps1')

function Assert-Equal {
    param($Expected, $Actual, [string]$Message)
    if ($Expected -ne $Actual) {
        throw "$Message Expected=[$Expected] Actual=[$Actual]"
    }
}

$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$testDirectory = Join-Path $tempRoot ('serial-recorder-check-' + [Guid]::NewGuid().ToString('N'))
$resolvedTestDirectory = [IO.Path]::GetFullPath($testDirectory)
if (-not $resolvedTestDirectory.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Refusing to create a test directory outside the system temp directory.'
}

try {
    Assert-Equal '"a,""b"""' (ConvertTo-CsvField 'a,"b"') 'CSV escaping failed.'
    Assert-Equal 'ok\u001B[31m' (ConvertTo-DisplayText "ok$([char]27)[31m") `
        'Control character escaping failed.'

    $state = New-LogState -Directory $resolvedTestDirectory -MaxBytes 1
    Write-LogLine -State $state -Source 'COM7' -Message '温度=24.6,状态="正常"' -QuietOutput
    Write-LogLine -State $state -Source 'COM7' -Message 'second line' -QuietOutput
    Close-LogState -State $state

    $files = @(Get-ChildItem -LiteralPath $resolvedTestDirectory -Filter '*.csv' | Sort-Object Name)
    Assert-Equal 2 $files.Count 'Log rotation failed.'
    $first = @(Import-Csv -LiteralPath $files[0].FullName -Encoding UTF8)
    $second = @(Import-Csv -LiteralPath $files[1].FullName -Encoding UTF8)
    Assert-Equal 1 $first.Count 'First log part has the wrong row count.'
    Assert-Equal 1 $second.Count 'Second log part has the wrong row count.'
    Assert-Equal '1' $first[0].sequence 'Sequence did not start at one.'
    Assert-Equal 'COM7' $first[0].source 'Source was not preserved.'
    Assert-Equal '温度=24.6,状态="正常"' $first[0].message 'UTF-8 or CSV content changed.'
    Assert-Equal '2' $second[0].sequence 'Sequence did not continue after rotation.'

    Write-Host 'CHECK_OK: CSV escaping, safe display, UTF-8 logging and rotation'
}
finally {
    if (Test-Path -LiteralPath $resolvedTestDirectory) {
        Remove-Item -LiteralPath $resolvedTestDirectory -Recurse -Force
    }
}
