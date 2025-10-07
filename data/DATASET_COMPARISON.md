# 📊 Complete Dataset Comparison Analysis
## Granite Guardian: Predictive Maintenance Advisor Project

**Project Duration**: September 2025 - May 2026  
**Partners**: IBM UK & UCL  
**Objective**: Use IBM Granite for OBD-II data analysis and predictive maintenance

---

## 🎯 Executive Summary

This project has access to **4 distinct OBD-II/CAN-BUS datasets** with varying characteristics:

| Dataset | Size | Files | Sampling Rate | Best For | Status |
|---------|------|-------|---------------|----------|--------|
| **carOBD** | ~135K rows | 131 files | 1 Hz | Production baseline | ✅ Recommended |
| **KIT** | 2.7M+ rows | 163 files | 10 Hz | ML training | ✅ Recommended |
| **HDTruck** | 140M+ rows | 2,800 files | High freq | Advanced R&D | ⚠️ Raw CAN |
| **CAN-BUS** | ~1K rows | 18 files | Low freq | Quick prototyping | ⚠️ Small sample |

---

## 📁 Dataset 1: carOBD (Toyota Etios 2014)

### Overview
- **Source**: GitHub - Master's thesis dataset
- **Vehicle**: Toyota Etios (2014, 1496 CC engine)
- **Hardware**: Carloop (OBD-II with cellular connectivity)
- **Total Files**: 131 CSV files
- **Sampling Rate**: 1 Hz (1 sample per second)
- **Total Rows**: ~135,000 data points

### Data Structure
```
27 OBD-II Parameters (PIDs):
✓ ENGINE_RPM, ENGINE_RUN_TIME, ENGINE_LOAD
✓ VEHICLE_SPEED, THROTTLE, COOLANT_TEMPERATURE
✓ INTAKE_MANIFOLD_PRESSURE, FUEL_TANK_LEVEL_INPUT
✓ CATALYST_TEMPERATURE (Bank 1, Sensors 1&2)
✓ CONTROL_MODULE_VOLTAGE, TIMING_ADVANCE
✓ Fuel trim (short/long term, Bank 1)
✓ Throttle positions (absolute, relative, commanded)
✓ Accelerator pedal positions (D, E)
✓ INTAKE_AIR_TEMPERATURE, ABSOLUTE_BAROMETRIC_PRESSURE
✓ Diagnostic codes: TIME_RUN_WITH_MIL_ON, DISTANCE_TRAVELED_WITH_MIL_ON
```

### File Categories
```
📂 drive*.csv (13 files)    - High-speed road driving
📂 idle*.csv (47 files)     - Engine on, vehicle parked (anomaly detection focus)
📂 live*.csv (39 files)     - Real commute (work → home)
📂 long*.csv (12 files)     - Long-distance trips
📂 ufpe*.csv (18 files)     - Low-speed campus driving
📂 Total: 131 files
```

### Sample Data
```csv
ENGINE_RPM, VEHICLE_SPEED, THROTTLE, COOLANT_TEMPERATURE, ENGINE_LOAD
0.0,        0.0,           17.6,     81.0,                 0.0
894.0,      0.0,           20.0,     79.0,                 55.3
1250.5,     25.0,          45.2,     85.0,                 68.4
```

### Strengths ✅
- **Clean, structured data** - Easy to parse and analyze
- **Diverse driving modes** - Idle, drive, long trips, campus
- **Real anomaly examples** - Includes actual sensor failures
- **Well-documented** - Published in IEEE paper
- **Consistent sampling** - Reliable 1 Hz rate
- **Production-ready format** - Direct OBD-II standard PIDs

### Limitations ⚠️
- Single vehicle type (passenger car only)
- Single geographic region (Brazil)
- Limited to 27 parameters (no advanced diagnostics)
- No DTC (Diagnostic Trouble Code) details in most files

### Use Cases for Project
1. **Baseline anomaly detection** (IDLE files with known sensor issues)
2. **Coolant temperature monitoring** (per IEEE paper focus)
3. **Real-world validation** of Granite predictions
4. **Diverse scenario testing** (idle vs drive vs long trip)

### Recommendation
**🟢 HIGHLY RECOMMENDED** - Best for initial development and validation. Use as primary dataset for:
- MoSCoW "Must": Detect anomalies in engine temperature, RPM, throttle
- Training Granite on realistic failure patterns
- Dashboard visualization with actual sensor trends

---

## 📁 Dataset 2: KIT (Karlsruhe Institute of Technology)

### Overview
- **Source**: FIZ Karlsruhe - Research repository
- **Total Rows**: 2,727,859 data points
- **Total Files**: 163 CSV files
- **Sampling Rate**: ~10 Hz (10 readings per second)
- **Vehicles**: Multiple vehicles (Seat Leon, VW, etc.)
- **Size**: Combined file ~500 MB

### Data Structure
```
Core Parameters (22 columns):
✓ timestamp, Time_original, unique_id, original_row_id
✓ Engine Coolant Temperature [°C]
✓ Intake Manifold Absolute Pressure [kPa]
✓ Engine RPM [RPM]
✓ Vehicle Speed Sensor [km/h]
✓ Intake Air Temperature [°C]
✓ Air Flow Rate from Mass Flow Sensor [g/s]
✓ Absolute Throttle Position [%]
✓ Ambient Air Temperature [°C]
✓ Accelerator Pedal Position D & E [%]

Metadata:
✓ source_file, date, vehicle
✓ route_from, route_to (RT, S, A, U, X routes)
✓ traffic_condition (Stau=traffic jam, Frei=free flow, Normal)
```

### Sample Data
```csv
timestamp,   Engine_Temp, RPM,   Speed, MAF,  Throttle, Traffic
1499238991,  31.0,        0.0,   0.0,   0.91, 89.0,     Stau
1499238991,  31.0,        894.0, 0.0,   2.15, 89.0,     Stau
1499238992,  32.0,        1250.0,15.0,  5.43, 45.0,     Normal
```

### Strengths ✅
- **Massive dataset** - 2.7M+ rows for robust ML training
- **High sampling rate** - 10 Hz captures rapid changes
- **Multiple vehicles** - Generalization across car models
- **Traffic context** - Labels for traffic jams vs free flow
- **Route information** - Geographic and usage pattern context
- **Real-world conditions** - Multiple drivers, routes, scenarios
- **Pre-cleaned** - Combined into single ready-to-use file
- **Metadata-rich** - Unique IDs, source tracking, date/time

### Limitations ⚠️
- European vehicles only (may differ from other regions)
- Limited to ~12 core OBD parameters (vs 27 in carOBD)
- No idle mode data (all driving scenarios)
- No explicit DTC codes or MIL status
- Missing: fuel system, catalyst temp, diagnostic codes

### Use Cases for Project
1. **ML model training** - Large dataset for deep learning
2. **Traffic-aware predictions** - Correlate faults with driving conditions
3. **Multi-vehicle generalization** - Test Granite across car types
4. **Time-series forecasting** - High-frequency data for trend prediction
5. **Route-specific analysis** - Predict failures based on usage patterns

### Recommendation
**🟢 HIGHLY RECOMMENDED** - Use as primary ML training dataset. Ideal for:
- Training robust anomaly detection models
- Validating Granite at scale
- Dashboard with time-series visualization (MoSCoW "Should")
- Testing multi-vehicle deployment scenarios

---

## 📁 Dataset 3: HDTruck (Finnish Heavy-Duty Truck)

### Overview
- **Source**: Finnish research institute (Etsin Fairdata)
- **Total Files**: 2,800 CSV files
- **Total Size**: 2.87 GB (compressed)
- **Recording Duration**: ~180 hours (7.5 days)
- **Rows per File**: ~50,000 CAN frames
- **Time per File**: ~1 minute
- **Sampling Rate**: Very high frequency (multiple messages per millisecond)

### Data Structure
```
Raw CAN Bus Format (4 columns):
timestamp;          id;            dlc; data
2020-11-23 08:03:31;0xcf003e6;     8;   255;255;255;255;255;255;255;255
2020-11-23 08:03:31;0x10ff80e6;    8;   255;255;255;47;255;255;255;255
2020-11-23 08:03:31;0x18f006e6;    8;   0;0;0;0;0;0;0;0

Fields:
- timestamp: Full datetime with microseconds
- id: CAN message identifier (hex)
- dlc: Data length code (bytes)
- data: 8 bytes of raw hex data (requires decoding)
```

### CAN Message IDs (Examples)
```
0xcf003e6      - Vehicle dynamics
0x10ff80e6     - Engine parameters
0x18f006e6     - Diagnostics
0x15ff59e6     - Transmission
... (hundreds of different message types)
```

### Strengths ✅
- **Massive scale** - 140M+ CAN frames, 180 hours
- **Heavy-duty vehicle** - Truck data (different failure modes)
- **Diagnostic codes** - Contains actual DTC messages
- **Real-world deployment** - Extended monitoring period
- **High resolution** - Captures millisecond-level events
- **Complete vehicle data** - Every CAN message logged

### Limitations ⚠️
- **Raw CAN format** - Requires DBC file for decoding
- **No OBD-II standard PIDs** - Heavy-duty trucks use J1939 protocol
- **Massive preprocessing needed** - Must decode CAN IDs manually
- **Storage intensive** - 2.87 GB compressed
- **Truck-specific** - Not directly comparable to passenger cars
- **Missing DBC file** - CAN message definitions not included

### Decoding Challenge
```python
# Example: Would need to decode like this
def decode_can_message(can_id, data):
    if can_id == 0xcf003e6:
        # Engine RPM = (data[3] * 256 + data[4]) * 0.125
        # Coolant temp = data[5] - 40
        pass  # Requires J1939 specification
```

### Use Cases for Project
1. **Advanced research** - If project scope expands to heavy vehicles
2. **Cross-domain validation** - Test Granite on truck vs car data
3. **Extended monitoring** - 180-hour baseline for long-term trends
4. **DTC analysis** - If you can decode diagnostic messages
5. **Publication novelty** - Unique heavy-duty dataset for ICLR

### Recommendation
**🟡 OPTIONAL / FUTURE WORK** - High value but requires significant effort:
- **Pros**: Unique heavy-duty data, DTC codes, massive scale
- **Cons**: Requires CAN decoding, J1939 expertise, preprocessing
- **Verdict**: Use only if:
  - You find/create DBC file for decoding
  - Project scope includes heavy-duty vehicles
  - You have time for extensive preprocessing
  - Conference paper needs novel heavy-vehicle angle

**Suggested approach**: Start with carOBD + KIT, add HDTruck later if time permits

---

## 📁 Dataset 4: CAN-BUS (Kaggle - Driving Datasets)

### Overview
- **Source**: Kaggle (Pakistan region)
- **Total Files**: 18 CSV files (organized into 5 categories)
- **Total Size**: ~580 KB
- **Total Rows**: ~1,000 data points
- **Sampling Rate**: Low frequency (~1 reading per minute)
- **Quality**: ⚠️ "This dataset is a catastrophe" (per README)

### Data Structure (Organized Categories)
```
1. Comprehensive Diagnostics (5 files, 479 KB)
   - 30+ OBD-II parameters (JSON format)
   - Full diagnostics: RPM, coolant, fuel, DTC codes
   - Files: OBD Reading.csv, obd-mobilehub*.csv

2. Engine Health (2 files, 63 KB)
   - Mass Air Flow (MAF) sensor data
   - GPS coordinates
   
3. Fuel Efficiency (1 file, 4 KB)
   - Fuel consumption, economy, CO2 emissions
   - 46 rows only

4. GPS Telemetry (4 files, 12 KB)
   - GPS routes (Phase 2 ↔ UET Campus)
   - Speed profiles
   
5. Raw Basic Trips (6 files, 11 KB)
   - Minimal GPS data for testing
```

### Sample Data (JSON Format)
```json
{
  "ENGINE_RPM": "894RPM",
  "ENGINE_COOLANT_TEMP": "79C",
  "SPEED": "0km/h",
  "THROTTLE_POS": "20.0%",
  "DTC_NUMBER": "MIL is OFF0 codes",
  "FUEL_TYPE": "Gasoline",
  "TIMING_ADVANCE": "50.6%"
}
```

### Strengths ✅
- **Well-organized** - Already categorized by use case
- **Comprehensive parameters** - 30+ PIDs in diagnostic files
- **DTC codes included** - Malfunction indicator lamp status
- **Documented** - Detailed README with use cases
- **Quick to load** - Small file sizes
- **Good for prototyping** - Fast iteration during development

### Limitations ⚠️
- **Tiny dataset** - Only ~1,000 total rows
- **Low sampling rate** - ~1 reading per minute (vs 1 Hz or 10 Hz)
- **Data quality issues** - Marked as "catastrophe" by curator
- **Inconsistent formatting** - Mix of JSON and CSV structures
- **Geographic limitation** - Single region (Pakistan)
- **Short duration** - Minutes of data, not hours/days
- **Not ML-ready** - Too small for training models

### Use Cases for Project
1. **Initial exploration** - Test data loading and parsing code
2. **UI prototyping** - Populate dashboard during development
3. **Demo/testing** - Quick validation of Granite API calls
4. **Edge cases** - JSON parsing challenges

### Recommendation
**🔴 NOT RECOMMENDED FOR PRODUCTION** - Use only for:
- Initial code development and testing
- Dashboard UI mockups (before loading real data)
- Quick Granite API validation

**Do NOT use for**:
- ML model training (too small)
- Anomaly detection validation (insufficient samples)
- Conference paper results (not credible)
- Real-world deployment testing

---

## 🎯 Recommended Dataset Strategy

### Phase 1: Development (October - December 2025)
**Primary**: carOBD dataset
- Start with IDLE files for anomaly detection
- Use drive files for normal behavior baseline
- Test all MoSCoW "Must" requirements:
  - ✅ Load and parse OBD-II time-series data
  - ✅ Detect anomalies in engine temp, RPM, throttle
  - ✅ Generate maintenance alerts with Granite

**Secondary**: CAN-BUS dataset
- Use comprehensive_diagnostics for UI prototyping
- Test JSON parsing and data normalization
- Quick Granite API integration tests

### Phase 2: ML Training (January - February 2026)
**Primary**: KIT dataset (2.7M rows)
- Train robust anomaly detection models
- 10 Hz sampling for accurate trend prediction
- Multi-vehicle generalization
- Traffic-aware predictive maintenance

**Validation**: carOBD dataset
- Test KIT-trained models on different vehicle
- Cross-dataset validation for paper credibility

### Phase 3: Dashboard & Visualization (March 2026)
**Combined**: carOBD + KIT
- Use carOBD for real anomaly examples
- Use KIT for historical trend visualization
- Implement MoSCoW "Should":
  - ✅ Visualize sensor trends over time
  - ✅ Dashboard for viewing alerts

### Phase 4 (Optional): Advanced Research (April - May 2026)
**If time permits**: HDTruck dataset
- Heavy-duty vehicle angle for novelty
- Extended monitoring (180 hours)
- DTC code analysis (if decoded)
- Potential ICLR paper differentiator

---

## 📊 Dataset Comparison Matrix

| Criteria | carOBD | KIT | HDTruck | CAN-BUS |
|----------|--------|-----|---------|---------|
| **Data Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Size** | 135K rows | 2.7M rows | 140M frames | 1K rows |
| **Sampling Rate** | 1 Hz | 10 Hz | >100 Hz | <1 Hz |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Documentation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Parameter Coverage** | 27 PIDs | 12 PIDs | 100+ CAN IDs | 30+ PIDs |
| **Anomaly Examples** | Yes (real) | No | Yes (DTCs) | Limited |
| **ML Training** | Good | Excellent | Excellent* | Poor |
| **Production Ready** | Yes | Yes | No* | No |
| **Vehicle Types** | Car | Cars | Truck | Car |
| **Geographic** | Brazil | Germany | Finland | Pakistan |
| **Preprocessing** | None | None | Heavy* | Moderate |

*Requires CAN decoding

---

## 🎓 Technical Specifications

### carOBD
```python
# Loading example
import pandas as pd

# Best for baseline
df = pd.read_csv('carOBD/obdiidata/idle1.csv')
# Columns: 27 OBD-II PIDs
# Rows: ~1000-3000 per file
# Memory: ~5-15 MB per file
```

### KIT
```python
# Loading example
df = pd.read_csv('KIT/dataset/combined_obd_dataset_watsonx.csv')
# Columns: 22 (12 sensors + 10 metadata)
# Rows: 2,727,859
# Memory: ~500 MB
# Features:
#   - traffic_condition (Stau/Normal/Frei)
#   - route_from, route_to
#   - vehicle type
#   - 10 Hz sampling
```

### HDTruck
```python
# Loading example (requires decoding)
df = pd.read_csv('HDTruck/part_1/20201123075441304067.csv', sep=';')
# Columns: 4 (timestamp, id, dlc, data)
# Rows: ~50,000 per file (2800 files)
# Memory: 2.87 GB total
# Challenge: Need DBC file to decode CAN IDs
```

### CAN-BUS
```python
# Loading example (JSON parsing required)
import json
df = pd.read_csv('CAN-BUS/organized/comprehensive_diagnostics/OBD Reading.csv')
# Column: "readings (M)" contains nested JSON
# Need to parse: json.loads(df['readings (M)'][0])
# Rows: ~100 per file
# Memory: ~1-2 MB per file
```

---

## 🏆 Final Recommendation for IBM/UCL Project

### Must Use (Priority 1)
1. **carOBD** - Primary development and validation dataset
   - Reason: Clean, diverse, real anomalies, production-ready
   - Use for: Anomaly detection, Granite training, dashboard demos

2. **KIT** - Primary ML training dataset
   - Reason: Large scale, high sampling rate, multi-vehicle
   - Use for: Model training, time-series forecasting, paper results

### Should Consider (Priority 2)
3. **HDTruck** - Advanced research (if time permits)
   - Reason: Unique heavy-duty data, long duration, DTCs
   - Use for: Paper novelty, cross-domain validation
   - Caveat: Requires CAN decoding effort

### Skip (Priority 3)
4. **CAN-BUS** - Prototyping only
   - Reason: Too small, low quality, inconsistent
   - Use for: Quick tests only
   - Avoid: Production use, paper results

---

## 📝 MoSCoW Requirements Mapping

### Must Have
| Requirement | Best Dataset |
|-------------|--------------|
| Load and parse OBD-II time-series data | carOBD (clean CSVs) |
| Detect anomalies in engine temperature | carOBD (idle files with known issues) |
| Detect anomalies in RPM | KIT (high sampling rate) |
| Detect anomalies in throttle | carOBD (diverse driving modes) |
| Use IBM Granite to generate alerts | Both carOBD + KIT |

### Should Have
| Requirement | Best Dataset |
|-------------|--------------|
| Visualize sensor trends over time | KIT (2.7M rows, 10 Hz) |
| Dashboard for viewing alerts | carOBD (real anomalies) |

### Could Have
| Requirement | Best Dataset |
|-------------|--------------|
| Integrate with chatbot for Q&A | KIT (metadata-rich) |

### Won't Have
| Requirement | Note |
|-------------|------|
| Connect to live vehicle data | As specified - not in scope |

---

## 📈 Success Metrics for ICLR Submission

### Dataset Requirements for Top-Tier Conference
1. **Scale**: ✅ KIT provides 2.7M rows
2. **Diversity**: ✅ carOBD provides multiple scenarios
3. **Real anomalies**: ✅ carOBD has documented failures
4. **Novelty**: ⭐ HDTruck adds heavy-duty angle
5. **Reproducibility**: ✅ All datasets are public

### Recommended Paper Structure
```
1. Introduction
   - Predictive maintenance challenge
   - IBM Granite for natural language diagnostics

2. Datasets
   - carOBD: Real-world anomaly validation (135K samples)
   - KIT: Large-scale training (2.7M samples)
   - (Optional) HDTruck: Heavy-duty generalization (140M frames)

3. Methodology
   - Time-series anomaly detection
   - Granite-based natural language generation
   - Multi-dataset validation

4. Results
   - Anomaly detection accuracy (carOBD)
   - Scalability validation (KIT)
   - Cross-domain generalization (HDTruck)

5. IBM Granite Integration
   - Natural language diagnostic generation
   - Maintenance recommendation system
```

---

## 🔗 Dataset Sources & Citations

### carOBD
- **Paper**: "Detecting Anomalies in the Engine Coolant Sensor Using One-Class Classifiers"
- **IEEE**: https://ieeexplore.ieee.org/abstract/document/8891367
- **GitHub**: https://github.com/eron93br/carOBD
- **Citation**: Required if used in publication

### KIT
- **Source**: FIZ Karlsruhe - Leibniz-Institut für Informationsinfrastruktur
- **License**: Apache 2.0
- **Metadata**: Included in dataset/descriptive-md/

### HDTruck
- **Source**: Etsin Fairdata (Finland)
- **URL**: https://etsin.fairdata.fi/dataset/7586f24f-c91b-41df-92af-283524de8b3e
- **Research URL**: https://research.fi/en/results/dataset/7586f24f-c91b-41df-92af-283524de8b3e

### CAN-BUS
- **Source**: Kaggle
- **URL**: https://www.kaggle.com/datasets/anwarmehmoodsohail/driving-datasets-obd-iican-bus
- **Quality Note**: "This dataset is a catastrophy" (per project notes)

---

## ✅ Action Plan

### Week 1 (Now)
- [x] Complete dataset comparison analysis
- [ ] Load carOBD idle files
- [ ] Implement basic OBD-II parser
- [ ] Test Granite API with sample data

### Week 2-4
- [ ] Train anomaly detection on carOBD
- [ ] Implement coolant temperature monitoring
- [ ] Generate first Granite maintenance alerts
- [ ] Build basic dashboard

### Month 2-3
- [ ] Scale to KIT dataset (2.7M rows)
- [ ] Train robust time-series models
- [ ] Multi-vehicle validation
- [ ] Advanced dashboard with trend visualization

### Month 4-5 (Optional)
- [ ] Decode HDTruck CAN data (if time permits)
- [ ] Heavy-duty vehicle validation
- [ ] Extended monitoring analysis

### Month 6-8
- [ ] ICLR paper writing
- [ ] IBM graduate software engineer interviews
- [ ] Final project delivery

---

**Document Created**: October 6, 2025  
**Last Updated**: October 6, 2025  
**Project**: Granite Guardian - Predictive Maintenance Advisor  
**Partners**: IBM UK & University College London

