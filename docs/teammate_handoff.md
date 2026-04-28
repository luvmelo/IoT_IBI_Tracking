# 队友需要发给你什么 / What teammate should send

## 你队友刚发给你的是什么？

是 mmWave Studio 的 **GUI state file**（一般叫 `mmWaveStudio.ini`，位于
`C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\` 之类的目录）。

它**只记录了**：
- mmWave Studio 版本（2.1.1.0）
- 用的是 IWR1443（`BoardType=1`）
- COM 端口是 3，波特率 921600
- 固件文件位置（`C:\ti\mmwave_studio_02_01_01_00\...`）
- 没在用 TDA 抓数据板（`TDA2xx_IPAddress=0.0.0.0`）→ 走 DCA1000，✅

它**不包含**：
- 任何 chirp / frame / ADC 配置
- 任何雷达数据
- 任何能让你在 Mac 上分析的东西

换句话说，对你做后续分析来说**这个 `.ini` 没什么用**——它只是确认了队友
那台 Windows 的环境是对的。

## 你真正需要队友发给你的（按重要性）

### 1. **必须** — Lua 配置脚本（`*.lua`，几 KB）

这是定义 chirp / frame / 采样率 / 帧大小的文件。**不知道这个，原始 `.bin`
没法解码**——你不知道一帧有多少 chirp、多少 sample、多少 RX 通道。

队友应该发你他们运行 mmWave Studio 时 `dofile()` 的那个 Lua 脚本（如果他们
用的是 [MakeyMakey 的 1443 配置](https://github.com/r-bt/MakeyMakey/blob/main/scripts/1443_mmwavestudio_config_old.lua)
就直接发那一份，我们也能直接用）。

放到 `data/` 目录下，比如 `data/capture_config.lua`。

### 2. **必须** — 录到的雷达数据。两种格式都行：

| 格式 | 来自哪里 | 优点 | 缺点 |
|---|---|---|---|
| **`.npz`** | MakeyMakey `record.py` | 已经 reshape 成 `(n_frames, n_chirps, n_samples, n_rx)` 复数；自带压缩；最适合 Mac 分析 | 需要队友先 clone MakeyMakey 并跑 `record.py` |
| **`adc_data.bin`** | mmWave Studio `CaptureCardConfig_StartRecord` 直接产出 | 不需要 Python，Lua 脚本默认就会保存到 `SAVE_DATA_PATH` | 是裸 int16 流，要靠 Lua 配置才能解码；文件可能很大（几百 MB+） |

**我已经把两种格式都支持了**，你哪个拿到就用哪个，但**优先要 `.npz`**：
- 文件小很多（压缩过）
- 已经 reshape 好，跨平台稳定（不会因为 endianness 等乱码）
- 如果队友的录制有掉包（DCA1000 偶尔会出这种问题），`.npz` 已经做过 zero-padding

### 3. **强烈建议** — 录制元信息

队友顺便发一条文字消息给你，写清楚：
- 录制时人在做什么（坐着 / 站着 / 距离雷达多少米）
- 录了多久（秒数）
- 雷达指向哪边
- 期间有没有 mmWave Studio 报错（"Packet drop" 之类）

这些信息你后面做 IBI/HRV 分析时一定会用到。

### 4. **可选** — `pktlogfile.txt`（如果他们用 mmWave Studio 直接录的）

这是 DCA1000 的丢包日志，调试用。如果数据看起来怪，发给我看一下能定位
是录制问题还是分析问题。

## 你可以直接复制粘贴发给队友的话

> 嘿，那个 `.ini` 是 mmWave Studio 自己存的 GUI 配置，里面没有 chirp 参数
> 也没有数据，我这边没法用 ¯\\\_(ツ)\_/¯
>
> 我需要这两个：
> 1. 你跑 mmWave Studio 时 `dofile()` 的那个 `.lua` 脚本（几 KB）
> 2. 录到的数据。**最好是 `.npz`**——按这个流程跑：
>
>    ```powershell
>    git clone https://github.com/r-bt/MakeyMakey.git
>    cd MakeyMakey
>    pip install numpy numba pyserial
>    # 先在 mmWave Studio 里 dofile() 你的 .lua 脚本，让雷达开始流数据
>    # 然后开新的 PowerShell：
>    python record.py --cfg path\to\your.lua
>    # 录够了 Ctrl+C，会在 data/ 下生成 radar_data_YYYYMMDD_HHMMSS.npz
>    ```
>
>    如果你嫌麻烦，直接把 mmWave Studio 录到的 `adc_data.bin` 发我也行（但
>    可能很大，先压缩一下）。
>
> 顺便告诉我：录的时候人坐哪、距离雷达多远、录了多少秒。

## 拿到文件以后你这边怎么跑

### 检查参数 + 看看数据正不正常

```bash
cd /Users/melo/Documents/Harvard_4/6.1820/Final_Project
uv run python scripts/inspect_capture.py \
    --cfg data/capture_config.lua \
    --data data/radar_data_xxx.npz   # 或 data/adc_data.bin
```

这会打印 range resolution、max range、frame size 等等，并在
`data/inspect_range_fft.png` 生成一张 range-FFT 图——能看到峰值就说明
数据正常。

### 在你自己代码里读

```python
from radar_analysis import RadarConfig, load_capture

cfg = RadarConfig("data/capture_config.lua")
params = cfg.get_params()
print(params)

data = load_capture("data/radar_data_xxx.npz", params)
# data: (n_frames, n_chirps, n_samples, n_rx) complex
```

### 已经验证过的（用合成数据跑了 round-trip 测试）

- Lua 解析 → 正确导出 n_chirps / n_samples / n_rx / range_res 等
- `.bin` 读回 → 形状正确，IQ 拼装正确
- `.npz` 读回 → 直接 unpack `data` key
- range FFT → 在合成 tone 的正确 bin 上有峰值

也就是说，队友数据一到你这边能直接跑。
