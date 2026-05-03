-- IWR1443 + DCA1000 capture script — revised from TI's stock
-- DataCaptureDemo_xWR.lua so that:
--   * the on-chip chirp test source is DISABLED. The stock demo enables
--     ar1.EnableTestSource(1) with two synthetic targets baked in via
--     ar1.SetTestSource(...), so the recorded .bin is the chip's
--     hardcoded fake pattern rather than real RF. Both calls are gone.
--   * chirp / frame / ADC parameters live as `KEY = value` at the top of
--     the file. The Mac-side analyzer (radar_analysis.RadarConfig)
--     regex-parses those exact keys (and `ar1.SelectChipVersion`,
--     `ar1.ChanNAdcConfig`, etc., anchored to column 0). All ar1.* calls
--     below are therefore unwrapped — the per-call `if (...) then
--     WriteToLog(...) end` boilerplate is removed because it indented
--     the calls and hid them from the parser. mmWave Studio's own
--     console still logs each API result.
--   * the post-StartFrame wait is computed from PERIODICITY × NUM_FRAMES
--     instead of a magic 5000 ms (the stock value was shorter than the
--     128 × 40 ms = 5.12 s capture, so it raced the recorder).
--   * StartMatlabPostProc is removed (TI's MATLAB post-proc toolbox does
--     not support IWR1443; analysis is done on the Mac side).
--   * the multi-chip (1642 / 1843 / 6843) branches are dropped. This
--     project is IWR1443-only (see CLAUDE.md). The script still reads
--     the part-ID register and warns if a different EVM is connected.

------------------------------------------------------------------
-- Capture parameters (parsed by radar_analysis.RadarConfig)
------------------------------------------------------------------
NUM_TX          = 1
NUM_RX          = 4

-- ProfileConfig (chirp shape)
START_FREQ      = 77       -- GHz
IDLE_TIME       = 100      -- us
RAMP_END_TIME   = 60       -- us
ADC_START_TIME  = 6        -- us
FREQ_SLOPE      = 29.982   -- MHz/us
ADC_SAMPLES     = 256
SAMPLE_RATE     = 10000    -- ksps  (10 MSPS)
RX_GAIN         = 30       -- dB

-- FrameConfig. END_CHIRP_TX = 0 keeps single-chirp frames, which is
-- what the Mac-side reshape (radar_analysis.reader) assumes. TDM-MIMO
-- is NOT supported by the current Python pipeline.
START_CHIRP_TX  = 0
END_CHIRP_TX    = 0
CHIRP_LOOPS     = 128      -- chirps per frame
NUM_FRAMES      = 128      -- 0 = continuous; otherwise stop after N frames
PERIODICITY     = 40       -- ms between frames

-- Post-StartFrame wait. For continuous streaming (NUM_FRAMES = 0),
-- default to 60 s and let the user re-run / Ctrl+C the recorder.
if (NUM_FRAMES == 0) then
    CAPTURE_MS = 60000
else
    CAPTURE_MS = NUM_FRAMES * PERIODICITY + 2000
end

------------------------------------------------------------------
-- mmWave Studio install + capture output paths.
--
-- Uses absolute paths so this .lua can live anywhere (Desktop, etc.)
-- and can be renamed freely. The stock demo derived paths from
-- debug.getinfo(1,'S').source, which only worked if the file was
-- named exactly "DataCaptureDemo_xWR.lua" AND located inside
-- mmWaveStudio\Scripts\. Both assumptions broke after the file was
-- moved/renamed and you got
--   "cannot open ...\<renamed>.lua\bitoperations.lua".
--
-- If your mmWave Studio install is not at the default location below,
-- change MMWAVE_STUDIO_PATH and TI_PATH to match.
------------------------------------------------------------------
TI_PATH           = "C:\\ti\\mmwave_studio_02_01_01_00"
MMWAVE_STUDIO_PATH = TI_PATH.."\\mmWaveStudio"
fw_path           = TI_PATH.."\\rf_eval_firmware"

-- Output .bin location. Make sure the directory exists before running;
-- mmWave Studio will not create it.
adc_data_path = MMWAVE_STUDIO_PATH.."\\PostProc\\adc_data.bin"

dofile(MMWAVE_STUDIO_PATH.."\\Scripts\\bitoperations.lua")

BSS_FW = fw_path.."\\radarss\\xwr12xx_xwr14xx_radarss.bin"
MSS_FW = fw_path.."\\masterss\\xwr12xx_xwr14xx_masterss.bin"

------------------------------------------------------------------
-- Sanity-check the connected EVM is actually an IWR1443. The other
-- xWR parts need different SAMPLE_RATE / LVDSLaneConfig /
-- CaptureCardConfig_Mode args; this script does not handle them.
------------------------------------------------------------------
res, efusedevice = ar1.ReadRegister(0xFFFFE214, 0, 31)
res, efuseES1device = ar1.ReadRegister(0xFFFFE210, 0, 31)
efuseES2ES3Device = bit_and(efusedevice, 0x03FC0000)
efuseES2ES3Device = bit_rshift(efuseES2ES3Device, 18)
if (efuseES2ES3Device == 0) then
    if (bit_and(efuseES1device, 3) == 1) then
        partId = 1443
    else
        partId = 0
    end
elseif (efuseES2ES3Device == 0xA0 or efuseES2ES3Device == 0x40) then
    partId = 1443
else
    partId = 0
end
if (partId ~= 1443) then
    WriteToLog("WARNING: this script targets IWR1443; detected efuse=" ..efuseES2ES3Device.. "\n", "red")
end

res, ESVersion = ar1.ReadRegister(0xFFFFE218, 0, 31)
ESVersion = bit_and(ESVersion, 15)

------------------------------------------------------------------
-- Bring up the chip
------------------------------------------------------------------
ar1.SelectChipVersion("XWR1443")
ar1.DownloadBSSFw(BSS_FW)
ar1.DownloadMSSFw(MSS_FW)
ar1.PowerOn(1, 1000, 0, 0)
ar1.RfEnable()

ar1.ChanNAdcConfig(1, 0, 0, 1, 1, 1, 1, 2, 1, 0)
ar1.LPModConfig(0, 0)
ar1.RfInit()
RSTD.Sleep(1000)

ar1.DataPathConfig(1, 1, 0)
ar1.LvdsClkConfig(1, 1)
ar1.LVDSLaneConfig(0, 1, 1, 1, 1, 1, 0, 0)

------------------------------------------------------------------
-- Profile / chirp / frame definition (uses KEY = value above)
------------------------------------------------------------------
ar1.ProfileConfig(0, START_FREQ, IDLE_TIME, ADC_START_TIME, RAMP_END_TIME, 0, 0, 0, 0, 0, 0, FREQ_SLOPE, 0, ADC_SAMPLES, SAMPLE_RATE, 0, 0, RX_GAIN)
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0)

-- Defensive: explicitly disable the on-chip test source. The stock demo
-- enabled it (with synthetic targets via ar1.SetTestSource), which would
-- overwrite real RF data.
--
-- DO NOT use ar1.EnableTestSource(0) here — RadarStudio's binding for
-- EnableTestSource(N) ignores N and always re-enables the test source.
-- (Verified empirically: a call to ar1.EnableTestSource(0) was observed
-- to log as "DisableTestSource(0)" followed by "EnableTestSource(1)",
-- leaving the test source ON.) The correct disable is DisableTestSource.
ar1.DisableTestSource(0)

ar1.FrameConfig(START_CHIRP_TX, END_CHIRP_TX, NUM_FRAMES, CHIRP_LOOPS, PERIODICITY, 0, 0, 1)

------------------------------------------------------------------
-- DCA1000 capture path
------------------------------------------------------------------
ar1.SelectCaptureDevice("DCA1000")
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 1, 1, 2, 3, 30)
ar1.CaptureCardConfig_PacketDelay(25)

ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
RSTD.Sleep(1000)

ar1.StartFrame()
RSTD.Sleep(CAPTURE_MS)
