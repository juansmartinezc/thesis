# 1er Análisis: Todas las Variables

## Variables temporales
- `year`
- `month`

## Variables categóricas
- `state_name`
- `county_name`

## Variables edáficas
- `ocd`
- `cec`
- `phh2o`
- `clay`
- `silt`
- `sand`

## Variables climáticas
- `TMAX`
- `TMIN`
- `TAVG`
- `PRCP`
- `RH2M`
- `WS10M`
- `SMS_-8`

# Posibles variables a conservar

- `year`
- `county_name`
- `state_name`
- `ocd`
- `RH2M`
- `silt`
- `SMS_-8`
- `clay`
- `phh2o`
- `TMAX`

---

# Análisis de Random Forest

### Variables descartadas:

- `sand`: redundante con `silt` y `clay`.
- `TMIN`: muy correlacionada con `TMAX`.
- `TAVG`: redundante con `TMAX` y `TMIN`.
- `cec`: poca importancia y baja correlación.
- `month`: casi sin aporte explicativo.
- `WS10M`: poca importancia, sin correlación fuerte.
- `PRCP`: baja importancia y correlación marginal.
