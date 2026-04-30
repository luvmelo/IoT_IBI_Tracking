# IBI / HRV 算法文献清单

本项目当前用到的每一阶段的算法 + 参数 + 对应文献 / 标准。
按流水线顺序组织，给队友扫一眼就能定位"我们在哪一步用了什么、为什么"。

> 想看比这份更厚的综述（包括我们没采用的方法），见
> [`mmwave_ibi_hrv_research_note.md`](../mmwave_ibi_hrv_research_note.md)。

---

## 当前流水线（一图）

```
原始 IQ ──► Range-FFT ──► 选胸口 range bin (功率 × 运动方差)
       ──► remove_dc + atan2 + np.unwrap   (TI vital-signs §2.3)
       ──► median 去趋势 + Hampel 去尖峰
       ──► motion-energy mask (高动作段标红)
       ──► 0.8 – 4.0 Hz Butterworth 带通 + filtfilt   (心跳带)
       ──► scipy.find_peaks (refractory + prominence) + 抛物线 sub-sample
       ──► IBI 序列  ──► 物理范围 + 局部中值 ±20 % 剔除
       ──► Task Force 1996 HRV：mean IBI / HR / SDNN / RMSSD / pNN50
```

VMD-based 心跳提取（mmHRV / Yang 2021 路线）已经为下一版预留接口，
当前**未实现**——见 §"未来工作"。

---

## Stage 1 — Range-FFT + 胸口 range bin 选择

**算法**：复 FFT 沿 fast-time；对每个 bin 在 0.3–1.5 m 距离窗口内
计算 (mean power across frames / chirps / RX) × (1 + α · 归一化相位方差)，
取 argmax。motion-variance 项让胸口（动）的 bin 战胜墙/桌（静）的 bin。

| 文献 | 用处 |
|---|---|
| [TI Vital Signs Lab v1.4 (xWR1443) Quick-Start Guide](https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/1023/vitalSignsLab_5F00_xwr1443_5F00_QuickStartGuide.pdf) | "在距离窗口内挑功率最大 bin" 准则的来源 |
| [TI mmWave Range/Velocity FAQ](https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/1050220/faq-computing-maximum-range-velocity-and-resolution-mmwave-system) | range_res / range_max 公式 |
| Yang et al., **mmHRV: Contactless Heart Rate Variability Monitoring Using mmWave Radar**, IEEE IoT Journal 2021 (DOI: 10.1109/JIOT.2021.3075167)  ·  [ICASSP 2021 PDF](https://xiaolu1263.github.io/files/2021-mmHRV.pdf) | motion-variance bin 评分思路 |

---

## Stage 2 — 相位提取 + DC 偏置 + 去趋势 + 去尖峰

**算法**：

- 相位：`np.unwrap(np.angle(z))`，π-discont 默认值
- 位移：`d = λ/(4π) · Δφ`，77 GHz 下系数 ≈ 0.31 mm/rad
- DC removal：取胸口 bin 复信号的 mean 减掉，避免静态杂波把 atan2
  工作点压偏（这是真实数据下 #1 经验性改进）
- 去趋势：median filter 2 s 窗
- 去尖峰：Hampel filter，半窗 `k_w = 3`，阈值 `n_σ = 3 × 1.4826 × MAD`

| 文献 | 用处 |
|---|---|
| [`numpy.unwrap` 文档](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html) | π-discont 默认参数选择 |
| TI mmWave Vital Signs Developer Guide §2.3 "DC Offset Correction" | DC 偏置必须在 atan2 之前减；我们的 `remove_dc()` 就是这个 |
| [MATLAB `hampel` 函数文档](https://www.mathworks.com/help/signal/ref/hampel.html) | 默认 k_w = 3, n_σ = 3 的来源 |
| Pearson et al., **Generalized Hampel Filters**, EURASIP J. Adv. Signal Process. 2016 — [Springer](https://link.springer.com/article/10.1186/s13634-016-0383-6) | Hampel 滤波在生理信号去尖峰的形式化 |

---

## Stage 3 — 呼吸 / 心跳分离（带通滤波）

**算法**：4 阶 Butterworth IIR + `scipy.signal.filtfilt`（零相位）

| 频段 | 范围 |
|---|---|
| 呼吸 | 0.10 – 0.60 Hz （6–36 breaths/min） |
| 心跳 | 0.80 – 4.00 Hz （48–240 BPM） |

| 文献 | 用处 |
|---|---|
| [TI Vital Signs Lab v1.2 User Guide](https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/1023/vitalSigns_5F00_lab_5F00_user_5F00_guide_5F00_v1.2UPDATE.pdf) | TI 推荐的两段 IIR 频段，我们直接沿用 |
| [`scipy.signal.butter` 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) | filter 实现 |
| 心率检测论文 [PMC 9104941](https://pmc.ncbi.nlm.nih.gov/articles/PMC9104941/) | 0.8–2 Hz 带通在 mmWave vital-signs 下的应用 |

> ⚠️ **已知坑**：呼吸 0.3 Hz 的 3 次谐波 ≈ 0.9 Hz，落在心跳带内。
> Stage 7（VMD 升级）会专门处理这个，当前 baseline 在快呼吸场景会有少量串扰。

---

## Stage 4 — 心跳峰值检测

**算法**：`scipy.signal.find_peaks`，参数：

- `distance = round(60 / max_bpm × fs)` （默认 max 200 BPM → 300 ms 不应期）
- `prominence = 0.5 × std(h)`
- 检测后做 3 点抛物线 sub-sample 精修，把时间分辨率从 ±1/(2 fs) 提到 ~5 ms

| 文献 | 用处 |
|---|---|
| [`scipy.signal.find_peaks` 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html) | 主函数 |
| PPG 心率检测综述 [PMC 8869811](https://pmc.ncbi.nlm.nih.gov/articles/PMC8869811/) | 200–300 ms 心动不应期的医学依据 |
| mmHRV (上面引用过) | radar 心跳信号 sub-sample 精修在 IBI 上的收益 |

---

## Stage 5 — IBI 质控（剔除伪峰 / 不应期违反）

**算法**：双重过滤，两道都过才保留：

1. 物理范围 `300 ≤ IBI ≤ 1500 ms`（即 40–200 BPM）
2. 局部中值偏差 `|IBI - median_local| / median_local ≤ 0.20`，
   `median_local` 用 5-interval 居中滚动

| 文献 | 用处 |
|---|---|
| [Kubios HRV 预处理文档](https://www.kubios.com/blog/preprocessing-of-hrv-data/) | 行业标准的 "medium" 阈值（绝对 250 ms）来源；我们用相对 20% |
| Marco Altini, [**Artifact Removal for PPG-Based HRV**](https://medium.com/swlh/artifact-removal-for-ppg-based-heart-rate-variability-hrv-analysis-5c7d08b6523a) | 20% 局部中值阈值的启发式规则 |

---

## Stage 6 — HRV 指标（**金标准**，队友先读这两篇）

**算法**（Task Force 1996 公式）：

```
mean_IBI  = (1 / N) Σ NN_i                                      [ms]
mean_HR   = 60000 / mean_IBI                                    [BPM]
SDNN      = sqrt[ (1/(N-1)) · Σ (NN_i - NN̄)²       ]            [ms]   ← 注意 N-1
RMSSD     = sqrt[ (1/(N-1)) · Σ (NN_{i+1} - NN_i)² ]            [ms]
pNN50     = 100% × #{ |NN_{i+1} - NN_i| > 50 }  /  (N-1)        [%]    ← 严格 >
```

| 文献 | 用处 |
|---|---|
| **Task Force of the European Society of Cardiology and NASPE**, *Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use*, **European Heart Journal 1996, 17(3): 354–381**  ·  [PDF](https://academic.oup.com/eurheartj/article-pdf/17/3/354/1312587/17-3-354.pdf) | 我们所有 HRV 公式 + N-1 分母 + 5 min 短时窗口的来源；HRV 领域的"圣经" |
| **Shaffer & Ginsberg**, *An Overview of HRV Metrics and Norms*, Frontiers in Public Health 2017  ·  [PMC 5624990](https://pmc.ncbi.nlm.nih.gov/articles/PMC5624990/) | 现代综述，比 1996 那篇好读，先读这个再翻 1996 |

> ⚠️ **小坑**：有些 Python HRV 库（早期 biosppy 等）用 N 而不是 N-1。
> 5 min × 70 BPM ≈ 350 个 NN，差异 ~0.3%，但跟参考软件比的时候要核对约定。

---

## Stage 7 — 路线图：VMD 心跳提取（当前**未实现**）

如果 baseline 在真实数据上误差太大（最可能的失败模式：呼吸谐波串扰），
下一步换成 mmHRV 同款的 VMD 方法替代当前 Stage 3 的 Butterworth。

| 文献 | 用处 |
|---|---|
| Yang et al. **mmHRV** (上面已引) | 我们升级路径的目标算法 |
| [PMC 12317012](https://pmc.ncbi.nlm.nih.gov/articles/PMC12317012/) | radar vital-signs 上的 VMD 参数 (K=4, α=2000) |
| [arXiv 2502.11042](https://arxiv.org/html/2502.11042v1) | 自适应 VMD，进一步提升 |
| [`vmdpy`](https://github.com/vrcarva/vmdpy) | Python 包 |
| [DR-MUSIC harmonic suppression, Sci. Reports 2024](https://www.nature.com/articles/s41598-024-77683-1) | 解决呼吸 3 次谐波串扰心跳带的问题 |

---

## 一句话定位

> **当前版本采用 TI 官方 vital-signs 流水线 (range-FFT → 相位 → 带通 → find_peaks) + Task Force 1996 HRV，IBI 质控用 Kubios 风格 (绝对范围 + 局部中值 20%)。心跳提取 baseline 是 Butterworth 带通；后续升级路径是 mmHRV 的 VMD 方法。**

---

## 给队友的"先读 3 篇"

如果时间紧，按这个顺序读：

1. **TI Vital Signs Lab v1.4 Quick-Start Guide**（30 min）
   → 理解 range-bin → 相位 → 带通 这套流程
2. **Task Force 1996 HRV Standards**（1 h，可跳节）
   → 理解 SDNN / RMSSD / pNN50 的精确定义和 5 min 窗口要求
3. **mmHRV 论文（IEEE IoT 2021）**（1 h）
   → 看 radar IBI 的当前 SOTA 用了什么、误差是多少（mean IBI MAE ≈ 3.68 ms）
