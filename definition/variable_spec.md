# 大氣模型變數符號規格文件 (Variable Symbol Specification)

## 1. 命名原則 (Naming Conventions)

### 1.1 物理定義描述 (Description)

- **格式**: 小寫字母，單字間以底線 `_` 連接。
- **單位**: 必須符合 **MKS 制**。
- **倍率記錄**: 若需使用非 MKS 制表達，需註記換算倍率。
  - 例如: `unit: 100 Pa`, `unit: 1 %` (對應 0.01)。

### 1.2 物理量符號 (Symbols)

- **長度**: 優先使用 **2 個字母** 之縮寫，盡量不超過 **4 個字母**。
- **大小寫**: 區分大小寫。
- **共通概念符號**:
  - `UU`: 相對網格座標的 X 分量風。
  - `VV`: 相對網格座標的 Y 分量風。
  - `UM`: 相對地球經緯度座標的 X 分量風 (東風為正)。
  - `VM`: 相對地球經緯度座標的 Y 分量風 (北風為正)。
  - `TT`: 溫度 (Temperature)。
  - `QV`: 水氣混和比 (Specific Humidity / Water Vapor Mixing Ratio)。
  - `PP`: 壓力 (Pressure)。
  - `HH`: 高度 (Height / Geopotential Height)。

## 2. 垂直座標描述 (Vertical Coordinate Indication)

採用標準格式: `[變數符號]_[座標指示碼][垂直位置]`

### 2.1 座標指示碼 (Coordinate Codes)

- **p (Pressure)**: 等壓座標 (Pressure levels)，數值單位預設為 Pa 或 hPa (需於 Metadata 註明)。
- **z (Geometric)**: 垂直幾何座標 (Height levels)，數值單位預設為 m。
- **e (Model/Hybrid)**: 模式座標或混合座標 (Sigma/Hybrid levels)。
- **對數氣壓**: 統一換算回氣壓 `p` 表達。

### 2.2 範例 (Examples)

- `TT_p1000`: 1000 hPa 的溫度。
- `UU_e1`: 模式第一層的 X 網格風速。
- `HH_z2`: 離地 2 公尺高度。

## 3. 結構分類 (Structure Classification)

### 3.1 靜態變數 (Static Variables)

紀錄地理資訊，如地形高度、土地利用、經緯度等。

- `HGT`: 地形高度 (Terrain Height)。
- `MSK`: 陸地遮罩 (Land Mask)。
- `LAT`: 緯度。
- `LON`: 經度。

### 3.2 垂直大氣結構 (Vertical Structure)

具有多層垂直分布的物理量。

- 範例: `TT`, `UU`, `VV`, `QV`。

### 3.3 水平層場結構 (Horizontal Layered Structure)

描述二維的結構分布，包含 Surface (無厚度) 或 Layered (有厚度但以單層代表) 的場。

- `SKT`: 皮膚溫度 (Skin Temperature)。
- `PSFC`: 地面壓力 (Surface Pressure)。
- `RAIN`: 降水量。

---

# NetCDF4 Metadata 規範套用

以下為預計產出變數的 Metadata 模板範例：

```json
{
    "TT": {
        "description": "air_temperature",
        "units": "K",
        "standard_name": "air_temperature",
        "long_name": "Temperature"
    },
    "UM": {
        "description": "eastward_wind",
        "units": "m s-1",
        "standard_name": "eastward_wind",
        "long_name": "Zonal Wind (Earth-relative)"
    },
    "PP_p850": {
        "description": "pressure_at_850hPa",
        "units": "Pa",
        "scale_factor": 100,
        "standard_name": "air_pressure",
        "long_name": "Pressure at 850hPa Level"
    }
}
```
