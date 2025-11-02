# Wear Index Report Generator for Routes

This script generates **wear index reports** for vehicle routes. Given origin and destination postcodes, it automatically generates a route and produces a detailed report of tyre wear.

---

## 🧠 Overview

The generated report contains:

- **Route distance**  
- **Total wear for 6 tyres**  
- **Average wear index per km**  
- **Equivalent tread depth wear per 1000 km** on each tyre  
- A **map of the route** with markers indicating where wear occurs  

The script simulates tyre wear based on vehicle dynamics and wear model for a single tyre.

---

## ⚙️ Script Workflow

The main script calls the `wear_index` function to perform the calculations. The workflow is as follows:

1. **Geocode Postcodes**  
   - Postcodes are parsed using the `geocode` function.

2. **Generate Route**  
   - Route is computed using the `compute_route` function.

3. **Calculate Coordinates and Distance**  
   - The `route_info_to_meter` function converts route data into coordinates and distances.

4. **Resample Route for Simulation**  
   - The `resample_curve_fixed_step` function ensures a constant speed simulation.

5. **Extract Wear-Relevant Sections**  
   - Turns and sections where wear is significant are identified using the `extract_sections` function.

6. **Simulate Wear Index**  
   - Wear for each section is simulated using the `stitch_simulation` function.

7. **Generate Report**  
   - Results are compiled into a report with the `generate_report` function.

---

## 🧩 Wear Calculation Models

The script supports two models for calculating wear on a single tyre:

1. **Empirical Model** – Based on slip angle and load from empirical measurements
2. **Surrogate Model** – Derived from a wear model based on tyre-road contact and local wear law

Vehicle dynamics are simulated for a **6-wheel semi-trailer** equipped with **385/65R22.5 tyres**.

---

## 📚 References

Liu, C. (2024). *Investigation of Severe Abrasive Truck Tyre Wear.* [doi.org/10.17863/CAM.111100](https://doi.org/10.17863/CAM.111100)  

---

## 🧩 Usage

To run the script:

```bash
python wear_index_report.py --origin <ORIGIN_POSTCODE> --destination <DESTINATION_POSTCODE>
