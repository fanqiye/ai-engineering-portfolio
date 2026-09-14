# Serial Field Recorder

面向 Windows 现场调试的串口日志记录器。它把设备每一行输出写入带时间戳的
CSV，在串口断开后自动重连，并按文件大小滚动分卷。适合开发板联调、老化测试、
小批量出厂验收和售后故障复现。

只依赖 Windows PowerShell 5.1 和系统自带的 .NET，不需要安装 Python、驱动库或
后台服务。

## 使用

查看可用串口：

```powershell
.\serial-recorder.ps1 -List
```

记录 COM3，波特率 115200：

```powershell
.\serial-recorder.ps1 -Port COM3 -BaudRate 115200
```

日志默认写入 `logs/`。每行包含本地时间、序号、来源和设备原文；写入后立即刷新，
即使设备或终端异常退出，已经收到的记录仍保留。串口断开后每 3 秒重试，按
`Ctrl+C` 停止。

常用参数：

```powershell
# 单次连接，断开后退出
.\serial-recorder.ps1 -Port COM5 -Once

# GB18030 文本、CRLF 行尾、每 50 MB 分卷
.\serial-recorder.ps1 -Port COM4 -EncodingName GB18030 -LineEnding CRLF -MaxFileSizeMB 50

# 不接硬件，回放示例数据并检查 CSV 输出
.\serial-recorder.ps1 -InputFile .\examples\sample-device-output.txt -OutputDirectory .\logs
```

输出示例：

```csv
timestamp,sequence,source,message
"2026-09-14T10:32:18.421+08:00",1,"COM3","temperature=24.6,humidity=51"
```

## 自检

```powershell
.\tests\test-serial-recorder.ps1
```

自检不需要串口硬件，覆盖 CSV 转义、控制字符显示、UTF-8 中文记录和日志分卷。

## 边界

本工具记录以 LF 或 CRLF 结尾的文本协议。二进制帧、无行尾的连续数据流或高带宽
采样应使用协议分析仪或专用采集程序。串口驱动仍由设备厂商提供。
