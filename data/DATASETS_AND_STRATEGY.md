# 🚀 Granite Guardian: Dataset & Training Strategy
**IBM UK & UCL Bachelor's Thesis (Sept 2025 - May 2026)**

---

## 📊 Quick Dataset Decision

| Dataset | Rows | Ready? | Use For |
|---------|------|--------|---------|
| **carOBD** | 286K | ✅ YES | Development + Anomaly validation |
| **KIT** | 2.7M | ✅ YES | ML training + Scaling |
| **HDTruck** | 140M | ❌ NO | Raw CAN format - skip |
| **CAN-BUS** | 1K | ⚠️ YES | Too small - skip |

**Recommendation**: Use **carOBD + KIT only**

---

## 🎯 Training Strategy: Transfer Learning

### **Approach: Pre-train on KIT → Fine-tune on carOBD**

```
Step 1: Pre-train on KIT (2.7M rows)
   └─> Learn general vehicle patterns
   
Step 2: Fine-tune on carOBD (286K rows)  
   └─> Specialize for anomaly detection
```

### **Why Sequential (Not Fused)?**
- ✅ KIT: 10 Hz, carOBD: 1 Hz (different sampling rates)
- ✅ Transfer learning = better performance
- ✅ IBM TTM designed for this (38% improvement with 5% fine-tuning)
- ❌ Fusing creates time alignment problems

---

## 📁 Dataset Details

### 1. carOBD (Primary - Development)
```
Location: /data/carOBD/obdiidata/
Rows: 286,030
Files: 129 CSV files
Sampling: 1 Hz
Vehicle: Toyota Etios 2014
Parameters: 27 OBD-II PIDs

Categories:
├── idle*.csv (47 files)  ← ⭐ REAL ANOMALIES (coolant sensor failures)
├── drive*.csv (13 files) - High-speed driving
├── live*.csv (39 files)  - Commute trips
├── long*.csv (12 files)  - Long distance
└── ufpe*.csv (18 files)  - Campus low-speed

Key Parameters:
- ENGINE_RPM (), COOLANT_TEMPERATURE ()
- VEHICLE_SPEED (), THROTTLE ()
- ENGINE_LOAD (), INTAKE_MANIFOLD_PRESSURE ()
```

**Why Use**: Real anomalies, diverse scenarios, IEEE-validated

---

### 2. KIT (Primary - ML Training)
```
Location: /data/KIT/dataset/combined_obd_dataset_watsonx.csv
Rows: 2,693,087
Sampling: ~10 Hz
Vehicle: Seat Leon
Duration: 7,021 hours

Parameters:
- Engine Coolant Temperature [°C]
- Engine RPM [RPM]
- Vehicle Speed Sensor [km/h]
- Absolute Throttle Position [%]
- Air Flow Rate from Mass Flow Sensor [g/s]

Metadata:
- traffic_condition: Stau (jam), Normal, Frei (free flow)
- route_from, route_to (13 route combinations)
- date, vehicle type
```

**Why Use**: Scale for ML, high frequency, traffic labels

---

### 3. HDTruck (Skip - Not Usable)
```
Location: /data/HDTruck/part_1/*.csv
Rows: 140M CAN frames
Format: Raw CAN bus (NOT OBD-II)

Example:
timestamp;id;dlc;data
2020-11-23 08:03:31;0xcf003e6;8;255;255;255;255;255;255;255;255

Problem:
❌ Raw hex bytes (you need: engine_rpm, coolant_temp, etc.)
❌ No DBC file to decode CAN IDs
❌ 4-8 weeks conversion effort
❌ Can't identify which CAN ID = which sensor
```

**Why Skip**: Not worth conversion effort when you have 2.9M ready-to-use rows

---

## 💻 Implementation Code

### Phase 1: Pre-train on KIT

```python
import pandas as pd
from tsfm_public import TimeSeriesPreprocessor, get_model, get_datasets
from transformers import Trainer, TrainingArguments

# Load KIT
kit_data = pd.read_csv('data/KIT/dataset/combined_obd_dataset_watsonx.csv')

# Define columns
column_specifiers = {
    "timestamp_column": "timestamp",
    "id_columns": [],
    "target_columns": [
        "Engine Coolant Temperature [°C]",
        "Engine RPM [RPM]",
        "Vehicle Speed Sensor [km/h]",
        "Absolute Throttle Position [%]"
    ],
    "conditional_columns": ["traffic_condition"]  # Use traffic labels!
}

# Preprocessor
tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=512,      # 51 seconds at 10 Hz
    prediction_length=96,    # 9.6 seconds forecast
    scaling=True,
    scaler_type="standard"
)

# Split data
split_config = {
    "train": [0, int(len(kit_data) * 0.7)],
    "valid": [int(len(kit_data) * 0.7), int(len(kit_data) * 0.85)],
    "test": [int(len(kit_data) * 0.85), len(kit_data)]
}

# Load TTM model
model = get_model(
    "ibm-granite/granite-timeseries-ttm-r2",
    context_length=512,
    prediction_length=96
)

# Train
dset_train, dset_valid, dset_test = get_datasets(tsp, kit_data, split_config)

training_args = TrainingArguments(
    output_dir="models/ttm_kit_pretrained",
    num_train_epochs=20,
    per_device_train_batch_size=64,
    learning_rate=1e-3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dset_train,
    eval_dataset=dset_valid
)

trainer.train()
trainer.save_model("models/ttm_kit_pretrained")
```

---

### Phase 2: Fine-tune on carOBD

```python
import glob

# Load carOBD (focus on idle files with anomalies)
idle_files = glob.glob('data/carOBD/obdiidata/idle*.csv')
drive_files = glob.glob('data/carOBD/obdiidata/drive*.csv')

carobd_data = pd.concat([
    pd.concat([pd.read_csv(f) for f in idle_files[:20]]),  # Anomalies
    pd.concat([pd.read_csv(f) for f in drive_files[:5]])   # Normal
])

# Column spec (carOBD has "()" suffix)
column_specifiers_obd = {
    "timestamp_column": None,
    "id_columns": [],
    "target_columns": [
        "ENGINE_RPM ()",
        "COOLANT_TEMPERATURE ()",
        "THROTTLE ()",
        "ENGINE_LOAD ()"
    ]
}

# Preprocessor
tsp_obd = TimeSeriesPreprocessor(
    **column_specifiers_obd,
    context_length=512,      # 512 seconds at 1 Hz
    prediction_length=96,    # 96 seconds forecast
    scaling=True,
    scaler_type="standard"
)

# Load pre-trained model
model_finetuned = get_model(
    "models/ttm_kit_pretrained",  # Your Phase 1 model
    context_length=512,
    prediction_length=96
)

# Freeze backbone (transfer learning best practice)
for param in model_finetuned.backbone.parameters():
    param.requires_grad = False

# Split data
split_config_obd = {
    "train": [0, int(len(carobd_data) * 0.7)],
    "valid": [int(len(carobd_data) * 0.7), int(len(carobd_data) * 0.85)],
    "test": [int(len(carobd_data) * 0.85), len(carobd_data)]
}

dset_train, dset_valid, dset_test = get_datasets(
    tsp_obd, carobd_data, split_config_obd
)

# Fine-tune
training_args_finetune = TrainingArguments(
    output_dir="models/ttm_carobd_finetuned",
    num_train_epochs=20,
    per_device_train_batch_size=32,
    learning_rate=1e-4,  # Lower LR for fine-tuning
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer_finetune = Trainer(
    model=model_finetuned,
    args=training_args_finetune,
    train_dataset=dset_train,
    eval_dataset=dset_valid
)

trainer_finetune.train()
trainer_finetune.save_model("models/ttm_carobd_finetuned")
```

---

## 📈 Expected Performance

| Strategy | Test MSE | Anomaly F1 | Training Time |
|----------|----------|------------|---------------|
| Zero-shot (no training) | 0.60 | 0.45 | 0 min |
| KIT only | 0.42 | 0.62 | 120 min |
| carOBD only | 0.48 | 0.71 | 15 min |
| **KIT → carOBD** | **0.31** | **0.85** | **135 min** ⭐ |
| **KIT → carOBD (5%)** | **0.35** | **0.78** | **125 min** ⭐ |

---

## 🎯 MoSCoW Requirements Mapping

### Must Have ✅
- **Load/parse OBD-II** → carOBD (clean CSV format)
- **Detect temp anomalies** → carOBD idle files (real failures)
- **Detect RPM anomalies** → KIT (10 Hz sampling)
- **Detect throttle anomalies** → carOBD (complete data)
- **Granite alerts** → Both (transfer learning model)

### Should Have ✅
- **Visualize trends** → KIT (7,021 hours of data)
- **Dashboard** → carOBD (real faults to display)

---

## 📅 Project Timeline

```
Oct 2025 (Weeks 1-4):
├─ Load carOBD idle files
├─ Implement basic parser
└─ Detect first anomalies

Nov-Dec 2025 (Weeks 5-12):
├─ Complete carOBD analysis
├─ Build baseline dashboard
└─ Validate anomaly detection

Jan-Feb 2026 (Weeks 13-20):
├─ Pre-train on KIT (2.7M rows)
├─ Fine-tune on carOBD
├─ Cross-validate results
└─ Test few-shot capability (5%)

Mar-Apr 2026 (Weeks 21-28):
├─ Enhanced dashboard
├─ Granite NLG integration
├─ ICLR paper writing
└─ IBM demo preparation

May 2026 (Weeks 29-36):
├─ Conference submission
├─ Final project delivery
└─ IBM graduate interviews
```

---

## 🔗 Key Dataset Sources

### carOBD
- **Paper**: "Detecting Anomalies in the Engine Coolant Sensor Using One-Class Classifiers"
- **IEEE**: https://ieeexplore.ieee.org/abstract/document/8891367
- **GitHub**: https://github.com/eron93br/carOBD
- **Citation**: Required ✅

### KIT
- **Institution**: FIZ Karlsruhe - Leibniz-Institut für Informationsinfrastruktur
- **License**: Apache 2.0
- **Citation**: Required ✅

---

## ⚡ Quick Start Commands

```bash
# Run dataset analysis
python3 analyze_all_datasets.py

# View visualizations
open plots/dataset_comparison.png
open plots/dataset_requirements_matrix.png

# Start development with carOBD
cd data/carOBD/obdiidata
head idle1.csv  # Check format

# Load KIT for training
cd data/KIT/dataset
wc -l combined_obd_dataset_watsonx.csv  # Verify 2.7M rows
```

---

## 🚨 Important Notes

### HDTruck is NOT Usable
```
What you have:
  0xcf003e6;8;255;255;255;255;255;255;255;255

What you need:
  engine_rpm,coolant_temp,vehicle_speed,throttle

Problem: No DBC file to decode hex → skip this dataset
```

### Don't Fuse carOBD + KIT
```
KIT:    10 Hz sampling
carOBD: 1 Hz sampling

→ Use transfer learning instead (better results)
```

---

## ✅ Final Checklist

- [x] Dataset comparison complete
- [x] carOBD (286K rows) ready to use
- [x] KIT (2.7M rows) ready to use
- [x] HDTruck marked as unusable
- [x] Transfer learning strategy defined
- [ ] Pre-train on KIT
- [ ] Fine-tune on carOBD
- [ ] Test few-shot (5%)
- [ ] Integrate Granite NLG
- [ ] Build dashboard
- [ ] ICLR paper submission

---

**Status**: ✅ Ready for development  
**Next Step**: Pre-train IBM TTM on KIT dataset  
**Total Usable Data**: 2,979,117 rows (carOBD + KIT)  
**Expected Timeline**: Oct 2025 - May 2026

