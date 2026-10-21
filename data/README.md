# Data Directory

This directory contains the primary Delhi historical weather dataset:
- `delhi_weather.csv`: 26,803 daily observations spanning from 1951-01-01 to 2024-06-19.

## Schema
- `_id`: Record identifier integer
- `Date`: Observation date (DD-MM-YYYY)
- `Rain`: Precipitation amount in mm ('Tr' represents trace rainfall < 0.1 mm)
- `Temp Max`: Maximum observed daily temperature in °C
- `Temp Min`: Minimum observed daily temperature in °C
- `Temp_diff`: Difference between daily maximum and minimum temperatures in °C
- `Has_rained`: Qualitative rain indicator ('YES' or 'NO')
- `Month`: Calendar month integer (1 to 12)
- `Year`: Calendar year integer (1951 to 2024)
- `Day`: Calendar day of month integer (1 to 31)
- `Week`: ISO week integer
- `month_name`: Month string name (January to December)
