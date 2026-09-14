[CmdletBinding(DefaultParameterSetName = 'Capture')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'List')]
    [switch]$List,

    [Parameter(ParameterSetName = 'Capture')]
    [ValidatePattern('^(?i:COM)\d+$')]
    [string]$Port,

    [Parameter(Mandatory = $true, ParameterSetName = 'Replay')]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$InputFile,

    [Parameter(ParameterSetName = 'Capture')]
    [ValidateRange(1, 4000000)]
    [int]$BaudRate = 115200,

    [Parameter(ParameterSetName = 'Capture')]
    [ValidateSet('LF', 'CRLF')]
    [string]$LineEnding = 'LF',

    [Parameter(ParameterSetName = 'Capture')]
    [ValidateRange(100, 60000)]
    [int]$ReadTimeoutMilliseconds = 1000,

    [Parameter(ParameterSetName = 'Capture')]
    [ValidateRange(1, 3600)]
    [int]$ReconnectSeconds = 3,

    [Parameter(ParameterSetName = 'Capture')]
    [switch]$Once,

    [Parameter(ParameterSetName = 'Capture')]
    [Parameter(ParameterSetName = 'Replay')]
    [ValidateNotNullOrEmpty()]
    [string]$EncodingName = 'UTF-8',

    [Parameter(ParameterSetName = 'Capture')]
    [Parameter(ParameterSetName = 'Replay')]
    [ValidateNotNullOrEmpty()]
    [string]$OutputDirectory = '.\logs',

    [Parameter(ParameterSetName = 'Capture')]
    [Parameter(ParameterSetName = 'Replay')]
    [ValidateRange(0.001, 10240)]
    [double]$MaxFileSizeMB = 10,

    [Parameter(ParameterSetName = 'Capture')]
    [Parameter(ParameterSetName = 'Replay')]
    [switch]$Quiet
)

Set-StrictMode -Version Latest

function ConvertTo-CsvField {
    param([AllowEmptyString()][string]$Value)
    return '"' + $Value.Replace('"', '""') + '"'
}

function ConvertTo-DisplayText {
    param([AllowEmptyString()][string]$Value)

    $builder = New-Object System.Text.StringBuilder
    foreach ($character in $Value.ToCharArray()) {
        if ([char]::IsControl($character)) {
            [void]$builder.Append(('\u{0:X4}' -f [int]$character))
        }
        else {
            [void]$builder.Append($character)
        }
    }
    return $builder.ToString()
}

function New-LogState {
    param([string]$Directory, [long]$MaxBytes)

    $created = New-Item -ItemType Directory -Path $Directory -Force -ErrorAction Stop
    return @{
        Directory   = $created.FullName
        MaxBytes    = $MaxBytes
        Part        = 0
        PartRecords = 0
        Sequence    = 0
        Session     = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        Writer      = $null
        Path        = $null
    }
}

function Open-LogPart {
    param([hashtable]$State)

    if ($null -ne $State.Writer) { $State.Writer.Dispose() }
    $State.Part++
    $State.PartRecords = 0
    $name = 'serial-{0}-part{1:D3}.csv' -f $State.Session, $State.Part
    $State.Path = Join-Path $State.Directory $name
    $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
    $State.Writer = New-Object System.IO.StreamWriter($State.Path, $false, $utf8WithoutBom)
    $State.Writer.AutoFlush = $true
    $State.Writer.WriteLine('timestamp,sequence,source,message')
    Write-Host "Writing $($State.Path)"
}

function Write-LogLine {
    param(
        [hashtable]$State,
        [string]$Source,
        [AllowEmptyString()][string]$Message,
        [switch]$QuietOutput
    )

    if (($null -eq $State.Writer) -or
        (($State.PartRecords -gt 0) -and ($State.Writer.BaseStream.Length -ge $State.MaxBytes))) {
        Open-LogPart -State $State
    }

    $State.Sequence++
    $State.PartRecords++
    $timestamp = [DateTimeOffset]::Now.ToString(
        'yyyy-MM-ddTHH:mm:ss.fffzzz',
        [Globalization.CultureInfo]::InvariantCulture
    )
    $row = @(
        (ConvertTo-CsvField $timestamp),
        $State.Sequence,
        (ConvertTo-CsvField $Source),
        (ConvertTo-CsvField $Message)
    ) -join ','
    $State.Writer.WriteLine($row)

    if (-not $QuietOutput) {
        Write-Host ('[{0}] {1}' -f $timestamp, (ConvertTo-DisplayText $Message))
    }
}

function Close-LogState {
    param([hashtable]$State)
    if ($null -ne $State.Writer) {
        $State.Writer.Dispose()
        $State.Writer = $null
    }
}

function Invoke-FileReplay {
    param(
        [string]$Path,
        [hashtable]$State,
        [System.Text.Encoding]$Encoding,
        [switch]$QuietOutput
    )

    $reader = New-Object System.IO.StreamReader($Path, $Encoding, $true)
    try {
        while (($line = $reader.ReadLine()) -ne $null) {
            Write-LogLine -State $State -Source ('file:' + [IO.Path]::GetFileName($Path)) `
                -Message $line -QuietOutput:$QuietOutput
        }
    }
    finally { $reader.Dispose() }
}

function Invoke-SerialCapture {
    param(
        [string]$PortName,
        [int]$Speed,
        [string]$Terminator,
        [int]$TimeoutMilliseconds,
        [int]$RetrySeconds,
        [System.Text.Encoding]$Encoding,
        [hashtable]$State,
        [switch]$SingleConnection,
        [switch]$QuietOutput
    )

    Add-Type -AssemblyName System.IO.Ports
    while ($true) {
        $serial = New-Object System.IO.Ports.SerialPort
        $serial.PortName = $PortName
        $serial.BaudRate = $Speed
        $serial.DataBits = 8
        $serial.Parity = [System.IO.Ports.Parity]::None
        $serial.StopBits = [System.IO.Ports.StopBits]::One
        $serial.Handshake = [System.IO.Ports.Handshake]::None
        $serial.Encoding = $Encoding
        $serial.NewLine = if ($Terminator -eq 'CRLF') { "`r`n" } else { "`n" }
        $serial.ReadTimeout = $TimeoutMilliseconds

        try { $serial.Open() }
        catch {
            $serial.Dispose()
            if ($SingleConnection) { throw }
            Write-Warning "$PortName unavailable: $($_.Exception.Message)"
            Start-Sleep -Seconds $RetrySeconds
            continue
        }

        Write-Host "Connected to $PortName at $Speed baud. Press Ctrl+C to stop."
        $disconnect = $null
        try {
            while ($true) {
                try { $line = $serial.ReadLine() }
                catch [System.TimeoutException] { continue }
                catch {
                    $disconnect = $_
                    break
                }
                Write-LogLine -State $State -Source $PortName -Message $line `
                    -QuietOutput:$QuietOutput
            }
        }
        finally {
            if ($serial.IsOpen) { $serial.Close() }
            $serial.Dispose()
        }

        if ($SingleConnection) {
            if ($null -ne $disconnect) { throw $disconnect }
            return
        }
        if ($null -ne $disconnect) {
            Write-Warning "$PortName disconnected: $($disconnect.Exception.Message)"
        }
        Start-Sleep -Seconds $RetrySeconds
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($PSCmdlet.ParameterSetName -eq 'List') {
        Add-Type -AssemblyName System.IO.Ports
        $ports = @([System.IO.Ports.SerialPort]::GetPortNames() | Sort-Object)
        if ($ports.Count -eq 0) { Write-Host 'No serial ports found.' }
        else { $ports }
        return
    }

    $encoding = [System.Text.Encoding]::GetEncoding($EncodingName)
    $maxBytes = [long]($MaxFileSizeMB * 1MB)
    $state = New-LogState -Directory $OutputDirectory -MaxBytes $maxBytes
    try {
        if ($PSCmdlet.ParameterSetName -eq 'Replay') {
            Invoke-FileReplay -Path $InputFile -State $state -Encoding $encoding -QuietOutput:$Quiet
        }
        else {
            if ([string]::IsNullOrWhiteSpace($Port)) {
                throw 'Specify -Port COM3, use -List, or use -InputFile for replay.'
            }
            Invoke-SerialCapture -PortName $Port -Speed $BaudRate -Terminator $LineEnding `
                -TimeoutMilliseconds $ReadTimeoutMilliseconds -RetrySeconds $ReconnectSeconds `
                -Encoding $encoding -State $state -SingleConnection:$Once -QuietOutput:$Quiet
        }
    }
    finally { Close-LogState -State $state }
}
