# ProjectCanary - Subsidence Early-Warning System

ProjectCanary (formerly Bhoomi-NETr) is an early-warning ground stability and subsidence monitoring system designed for surface terrain above abandoned underground coal workings. The system combines distributed surface sensor nodes, an edge analysis engine, and a live web console to detect ground displacement, tilt anomalies, and high-frequency acoustic vibrations before structural failure occurs.

---

## Problem and Approach

Ground above historical and abandoned coal workings is subject to delayed, unpredictable subsidence sinkholes and trough collapses. Existing monitoring setups suffer from three main drawbacks:
1. Single sensor nodes go blind when a component or power source fails, creating dangerous false-safe assumptions.
2. Chained LoRa topology breaks whenever any intermediate relay node fails, dropping telemetry from all subsequent nodes down the line.
3. Dense uniform sensor grids are prohibitively expensive to install, power, and maintain across wide rural or peri-mining tracts.

ProjectCanary addresses these issues through a structured "chunk" topology and explainable data analytics:
- **Chunk Geometry**: One central LiDAR hub monitors four low-cost corner nodes arranged in a square.
- **Resilient Redundancy**: Optical line-of-sight distance measurements from the hub cross-verify ground movement independently of corner node health.
- **Sensor-Health-Aware Alerts**: The processing engine continuously assesses sensor health and telemetry confidence, clearly distinguishing genuine subsidence motion from sensor faults or transducer dropouts.

---

## Node Types

ProjectCanary defines three specialized node tiers and a modular deployment layout:

### Node 1 (Surface Corner Node)
- **Microcontroller**: ESP32 with integrated wireless connectivity.
- **Telemetry Link**: LoRa radio transceiver for long-range, low-power telemetry transmission.
- **Sensors**: GPS module for spatial localization, MPU6050 6-axis accelerometer/gyroscope for tilt monitoring, and ADS1115 with piezo-electric transducer for high-frequency acoustic vibration and micro-crack detection.
- **Power**: Solar panel paired with rechargeable battery storage.
- **Displacement**: No displacement sensor in production units. The HC-SR04 ultrasonic distance sensor is only a prototype stand-in during laboratory bench tests.

### Node 2 (LiDAR Hub)
- **Core Unit**: Built on the Node 1 base board with full LoRa communication.
- **Optical Ranging**: Single-point LiDAR (Benewake TFmini-i) mounted on a precision rotating mount.
- **Target Tracking**: Scans reflector plates mounted on surrounding Node 1 corner units to measure millimeter-scale relative surface displacement across the chunk.

### Node 3 (Underground Sensor Node)
- **Connectivity**: Rugged cable-connected umbilical link running down into underground cavities or mine voids.
- **Sensors**: Multi-gas telemetry (including CH4/CO) and O2 depletion sensors.
- **Housing**: Certified explosion-proof, moisture-sealed enclosure rated for hazardous underground coal mine environments.

### Chunk Layout and Spacing
- **Chunk Layout**: One Node 2 LiDAR hub positioned at the center, sighting four Node 1 corner units placed at square perimeter corners.
- **Spacing Strategy**: Spaced chunks deployed across broad areas by default; tiled edge-to-edge chunks concentrated in high-risk zones.
- **Planning Cost**: About Rs 25k per chunk (planning estimate, quotes pending).

---

## System Architecture

```text
  [ Underground ]
   +---------------------------------------+
   | Node 3 (Gas & O2 Sensors, Enclosure)  |
   +---------------------------------------+
                       |
               Industrial Cable
                       |
  [ Surface Chunk ]    v
   +---------------------------------------+
   | Corner Nodes (Node 1 x 4)             |
   | [ESP32 + Tilt + Piezo + GPS + Solar]  |
   +---------------------------------------+
          |                       ^
     LoRa |          Laser Sighting
          v                       |
   +---------------------------------------+
   | Central LiDAR Hub (Node 2)            |
   | [Benewake TFmini-i on Rotating Mount] |
   +---------------------------------------+
                       |
                   LoRa Link
                       |
                       v
   +---------------------------------------+
   | Gateway Node (Local Emergency Siren)  |
   +---------------------------------------+
                       |
               Cellular / Wi-Fi
                       |
                       v
   +---------------------------------------+
   | Supabase Cloud DB & Edge Ingestion    |
   +---------------------------------------+
                       |
          Realtime Streaming / Polling
                       |
         +-------------+-------------+
         |                           |
         v                           v
   +-------------------+   +--------------------+
   | Analysis Engine   |   | Web Dashboard      |
   | (Risk & Health)   |   | (React 19 + Vite)  |
   +-------------------+   +--------------------+
         |                           |
         v                           v
   +-------------------+   +--------------------+
   | SMS Alerts to     |   | Operator Control   |
   | Security Guards   |   | Room Displays      |
   +-------------------+   +--------------------+
```

---

## Implementation Status

| Component | Status | Details |
|---|---|---|
| ESP32 Node Posting to Dashboard | TESTED on real hardware | Physical ESP32 hardware node pushing live telemetry over Wi-Fi/HTTP directly to the Supabase Edge Function and live React console. |
| Piezo Vibration Sensing | TESTED on real hardware | Physical ADS1115 ADC reading analog piezo-electric acoustic shock and vibration signals. |
| Sensor Status Fault Flag | TESTED on real hardware | Transducer health flags (`mpu6050_status`, `hc_sr04_status`, `ads1115_status`) evaluated and ingested by the pipeline. Note: A generic `sensor_status` field is not yet in the active schema. |
| LoRa Link Between Two Modules | TESTED on real hardware | Point-to-point radio communication link validated between two LoRa physical transceivers. |
| SMS Alert to Phone | TESTED on real hardware | Emergency SMS dispatch verified with real phone delivery via Fast2SMS (tested by the team, live verification script in `test_real_sms.js`, automated suite in `tests/test_sms.js`). |
| Subsidence Risk Index | TESTED in software with simulated data | 0-100 risk scoring engine in Python (`engine/risk.py`) and TypeScript (`src/services/analysisAdapter.ts`). |
| Sensor Health & Confidence Engine | TESTED in software with simulated data | Automated confidence scoring (0-100%), stuck-sensor detection, and transducer fault handling (`engine/sensor_health.py`). |
| Trend Analysis & Projection | TESTED in software with simulated data | Velocity, acceleration, and time-to-threshold projections (`src/services/trendAnalysis.ts`, `tests/trend_analysis.test.ts`). |
| Threshold Alerts & Deduplication | TESTED in software with simulated data | Multi-tier threshold evaluation, state-transition deduplication, and cooldown gating (`engine/thresholds.py`, `supabase/functions/sensor-data/sms.ts`). |
| Dashboard Pages | TESTED in software with simulated data | Full interactive console across all primary views: Live Dashboard, Trend Analysis, Crack Detection, Risk Assessment, Structural Health, and Alerts. |
| LiDAR Hub (Rotating Mount) | PLANNED | Central Node 2 with rotating motorized mount and Benewake TFmini-i optical distance ranging to corner reflector plates. |
| Node 3 (Underground Gas & O2) | PLANNED | Cable-connected, hazardous-environment certified node with multi-gas and oxygen depletion detection. |
| Gateway Siren & Staged Evacuation | PLANNED | On-site gateway hardware siren and multi-stage evacuation escalation protocol requiring human operator confirmation. |
| Blast / Impact vs Subsidence Classifier | PLANNED | Machine learning or signal-processing classifier to separate transient blasting vibrations from true ground subsidence events. |
| Synthetic Event Generator | PLANNED | Software simulator tool for generating complex structural failure geometries and propagation waves. |
| Zone Risk Map | PLANNED | 2D/3D geospatial GIS zone map visualizing chunks, terrain risk contouring, and active evacuation perimeters. |
| Full LoRa-to-Cloud Production Network | PLANNED | Field-deployed mesh/star LoRa network routing through an on-site cellular gateway to the cloud backend. |

---

## Quick Start

### 1. Running with Fake Data (Simulator)

No physical hardware is required to evaluate the dashboard and ingestion pipeline.

```bash
# Terminal 1: Start the dashboard
npm run dev

# Terminal 2: Run the telemetry simulator in desired mode
node simulator.js                                    # Normal multi-node baseline drift
node simulator.js --scenario=excessive_tilt          # Ramp tilt beyond 25 degrees
node simulator.js --scenario=high_vibration         # Ramp vibration beyond 1.5g
node simulator.js --scenario=critical               # Critical tilt, vibration, and displacement
node simulator.js --scenario=critical --node=NODE_01 # Target a specific node
```

Open your browser at `http://localhost:5173`.

### 2. Running with Real Hardware

Physical sensor nodes transmit telemetry payloads via HTTP POST to the live Supabase Edge Function:
```http
POST https://<your-project-ref>.supabase.co/functions/v1/sensor-data
Content-Type: application/json
Authorization: Bearer <SUPABASE_ANON_KEY>
apikey: <SUPABASE_ANON_KEY>
```

Refer to [API.md](file:///c:/Users/HP/Downloads/Bhoomi-NETr/API.md) for full payload requirements and constraints. In addition to numeric readings, firmware sends explicit per-sensor hardware health flags:
- `mpu6050_status`: required `"ok"` or `"fault"`
- `hc_sr04_status`: required `"ok"` or `"fault"`
- `ads1115_status`: optional `"ok"` or `"fault"`

*(Note: The ingestion validator accepts only these explicit transducer flags; a generic `sensor_status` field is not accepted).*

### 3. Running the Standalone Analysis Engine and Pytest Suite

The core Python analytical engine can be executed independently from command line:

```bash
# Evaluate a sample reading JSON file
python run_analysis.py sample_data/sample_reading.json

# Run Python pytest test suite
pytest tests/

# Run TypeScript/Node test suites
npm run test:trends
npm run test:sms
```

---

## Data Contract and Dashboard Pages

### Ingestion Data Contract (JSON Payload)

The ingestion endpoint accepts the following JSON fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `node_id` | string | Yes | Unique node identifier (e.g., `"NODE_01"`) |
| `timestamp` | string | No | ISO 8601 UTC timestamp (defaults to server `now()`) |
| `tilt_x` | number | Yes | X-axis inclination angle in degrees |
| `tilt_y` | number | Yes | Y-axis inclination angle in degrees |
| `vibration` | number | Yes | Peak acceleration magnitude in g |
| `distance` | number | Yes | Distance reading in cm (prototype ultrasonic stand-in) |
| `mpu6050_status` | string | Yes | Hardware transducer status: `"ok"` or `"fault"` |
| `hc_sr04_status` | string | Yes | Hardware transducer status: `"ok"` or `"fault"` |
| `ads1115_status` | string | No | ADC transducer status: `"ok"` or `"fault"` |
| `piezo_peak` | number | No | Peak acoustic voltage from piezo transducer |
| `piezo_rms` | number | No | RMS acoustic voltage from piezo transducer |
| `piezo_peak_to_peak` | number | No | Peak-to-peak acoustic voltage |

### Dashboard Pages

The web console provides seven dedicated views:
- `/` (`Home`): System introduction, architecture walkthrough, platform services, and technical FAQ.
- `/dashboard` (`Dashboard`): Live overview of all active nodes, latest readings, risk badges, and summary metrics.
- `/structural-health` (`StructuralHealth`): Physical safety envelope visualization, tilt vs vibration scatter analysis, and threshold limit indicators.
- `/risk-score` (`RiskScore`): Detailed breakdown of the 0-100 Subsidence Risk Index and contributing factor weights.
- `/node/:nodeId` (`NodeDetail`): Deep-dive telemetry inspection and time-series telemetry charts for an individual node.
- `/trend-analysis` (`TrendAnalysis`): Kinematic velocity, acceleration calculations, and predictive failure projection curves.
- `/alerts` (`Alerts`): Historical emergency alert log, acknowledged notifications, and dispatch status audit.

---

## Known Limitations

- **Demonstration-Only Thresholds**: Current warning limits (`tilt >= 15 deg`, `vibration >= 1.0g`, `displacement >= 2.0cm`) are demonstration values designed to validate pipeline mechanics and must not be treated as certified mine-safety limits.
- **Thresholds Duplicated in TypeScript and Python**: Threshold limits and risk scoring formulas are maintained separately in the Python engine (`engine/thresholds.py`, `engine/risk.py`) and the TypeScript frontend adapter (`src/services/analysisAdapter.ts`), requiring manual synchronization.
- **Row Level Security (RLS) Off**: Row Level Security is disabled on the `sensor_readings` database table during Phase 1 prototype development.
- **Latency Measured Only with a Simulated Node**: The sub-second streaming latency figure was measured using the software simulator running locally against the cloud database. Field latency over LoRa and cellular gateways remains uncharacterized.
- **No Real Collapse Data Yet**: The analysis model and alert triggers have not yet been evaluated against true geological ground collapse or active mine subsidence events.
- **HC-SR04 is a Stand-in**: The ultrasonic HC-SR04 distance sensor is used strictly as a bench prototype stand-in and will not be deployed in production field nodes.

---

## Team Roles & Hardware Repository

### Team Roles and Responsibilities
- **Person 1 (Data / Backend)**: Supabase PostgreSQL schema, Edge Function ingestion pipeline, hardware simulator, and database verification.
- **Person 2 (Analysis / AI)**: Sensor processing engine, ground displacement calculation, threshold logic, and automated risk scoring.
- **Person 3 (Frontend / Dashboard)**: Live monitoring console, multi-node status views, real-time WebSocket subscriptions, and time-series telemetry charts.

### Hardware Repository
- Hardware design schematics, wiring diagrams, and ESP32 firmware source code: [Bhoomi-NETr Hardware Repository](https://github.com/itskavyansh/Bhoomi-NETr.git) (repo name predates the rename).
