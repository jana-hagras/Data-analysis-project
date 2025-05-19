--1. What are the most common accident severity levels?
SELECT Severity_Encoded, COUNT(*) as accident_count
FROM car_accident
GROUP BY Severity_Encoded
ORDER BY accident_count DESC;

--2. Which days of the week have the highest accident frequency?
SELECT Day_of_week, COUNT(*) as accidents
FROM car_accident
GROUP BY day_of_week
ORDER BY accidents DESC;

--3. How do accidents vary between urban and rural areas?
SELECT Urban_or_Rural_Area, 
       COUNT(*) as total_accidents,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage
FROM car_accident
GROUP BY Urban_or_Rural_Area;

--4. What are the top 5 most dangerous road surfaces?
SELECT TOP 5 Road_Surface_Conditions, 
              Severity_Encoded, 
              COUNT(*) as accidents
FROM car_accident
GROUP BY Road_Surface_Conditions, Severity_Encoded
ORDER BY accidents DESC;

--5. Which vehicle types are most frequently involved in severe accidents?
SELECT TOP 5 vehicle_type, COUNT(*) AS severe_accidents
FROM car_accident
WHERE Accident_Severity IN ('Serious', 'Fatal')
GROUP BY vehicle_type
ORDER BY severe_accidents DESC;


--6. How does accident frequency change by month/season?
SELECT  Month, 
       COUNT(*) AS accidents
FROM car_accident
GROUP BY month
ORDER BY month;

--7. What’s the distribution of accidents by lighting conditions?
SELECT light_conditions, 
       COUNT(*) as accidents,
       ROUND(AVG(severity_encoded), 2) as avg_severity
FROM car_accident
GROUP BY light_conditions
ORDER BY accidents DESC;


--8. What’s the correlation between driver age and accident severity?
SELECT Age_band_of_driver, 
       Severity_Encoded,
       COUNT(*) as accidents
FROM car_accident
GROUP BY Age_band_of_driver, Severity_Encoded
ORDER BY Age_band_of_driver, accidents DESC;

--9. How does weather impact accident severity?
SELECT Weather_Conditions , 
       Severity_Encoded,
       COUNT(*) as accidents
FROM car_accident 
WHERE Weather_Conditions != 'Unknown'
GROUP BY Weather_Conditions, Severity_Encoded
ORDER BY accidents DESC;
use Accident;

--10. Which districts have the highest accident severity rates?
WITH district_stats AS (
    SELECT [District Area], Accident_Severity, COUNT(*) AS accident_count
    FROM car_accident
    WHERE [District Area] IS NOT NULL
    GROUP BY [District Area], Accident_Severity
)
SELECT  [District Area], Accident_Severity, accident_count,
    CAST(ROUND(100.0 * accident_count / SUM(accident_count) OVER (PARTITION BY [District Area]), 2) AS DECIMAL(5,2)) AS severity_percentage
FROM district_stats
ORDER BY [District Area], accident_count DESC;

--11. During which hours of the day do most severe accidents occur?
SELECT DATEPART(HOUR, Time) AS hour_of_day, Accident_Severity, COUNT(*) AS accident_count,
    RANK() OVER (PARTITION BY Accident_Severity ORDER BY COUNT(*) DESC) AS hour_rank
FROM car_accident
GROUP BY DATEPART(HOUR, Time), Accident_Severity
ORDER BY Accident_Severity, accident_count DESC;


--12. Which age and gender groups are most at risk for severe accidents?
SELECT Age_band_of_driver, Sex_of_driver, COUNT(*) AS total_accidents,
    CAST(AVG(Severity_Encoded) AS DECIMAL(3,1)) AS avg_severity,
    CAST(ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS DECIMAL(5,2)) AS percentage_of_total
FROM car_accident
WHERE Age_band_of_driver IS NOT NULL
GROUP BY Age_band_of_driver, Sex_of_driver
ORDER BY avg_severity DESC;

--13. What types of collisions occur most frequently in different weather conditions?
SELECT  Weather_Conditions, Type_of_collision, COUNT(*) AS accident_count,
    CAST(AVG(Severity_Encoded) AS DECIMAL(3,1)) AS avg_severity,
    CAST(ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY Weather_Conditions), 2) AS DECIMAL(5,2)) AS weather_percentage
FROM car_accident
WHERE Weather_Conditions NOT IN ('Unknown', 'Other')
GROUP BY Weather_Conditions, Type_of_collision
ORDER BY Weather_Conditions, accident_count DESC;

--14. How does accident severity vary between urban and rural areas?
SELECT Urban_or_Rural_Area, Accident_Severity, COUNT(*) AS accident_count,
    CAST(ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY Urban_or_Rural_Area), 2) AS DECIMAL(5,2)) AS area_percentage
FROM car_accident
WHERE Urban_or_Rural_Area IS NOT NULL
GROUP BY Urban_or_Rural_Area, Accident_Severity
ORDER BY Urban_or_Rural_Area, accident_count DESC;

--15. Which vehicle types are most frequently involved in the most severe accidents?
SELECT Vehicle_Type,COUNT(*) AS total_accidents, CAST(AVG(Severity_Encoded) AS DECIMAL(3,1)) AS avg_severity,
    SUM(CASE WHEN Accident_Severity = 'Fatal' THEN 1 ELSE 0 END) AS fatal_count
FROM car_accident
WHERE Vehicle_Type IS NOT NULL
GROUP BY Vehicle_Type
HAVING COUNT(*) > 50
ORDER BY avg_severity DESC;

--16. What are the most common causes of accidents for different driver experience levels?
SELECT Driving_experience, Cause_of_accident, COUNT(*) AS accident_count,
    CAST(ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY Driving_experience), 2) AS DECIMAL(5,2)) AS cause_percentage
FROM car_accident
WHERE Driving_experience IS NOT NULL AND Cause_of_accident IS NOT NULL
GROUP BY Driving_experience, Cause_of_accident
ORDER BY Driving_experience, accident_count DESC;

--17. How do junction types affect accident severity in different road conditions?
SELECT Types_of_Junction, Road_Surface_Conditions, CAST(AVG(Severity_Encoded) AS DECIMAL(3,1)) AS avg_severity,
		COUNT(*) AS accident_count
FROM car_accident
WHERE Types_of_Junction != 'No junction' AND Road_Surface_Conditions IS NOT NULL
GROUP BY Types_of_Junction, Road_Surface_Conditions
ORDER BY avg_severity DESC;


--18. What's the accident cause correlation matrix?  (What are the correlation patterns between different accident causes?)
WITH cause_stats AS (
    SELECT Cause_of_accident AS accident_cause, COUNT(*) AS frequency, AVG(CAST(Severity_Encoded AS FLOAT)) AS avg_severity
    FROM car_accident
    WHERE Cause_of_accident IS NOT NULL
    GROUP BY Cause_of_accident
    HAVING COUNT(*) > 5
),
cause_pairs AS (
    SELECT a.accident_cause AS cause1,  b.accident_cause AS cause2, a.avg_severity AS avg_severity1,
        b.avg_severity AS avg_severity2, a.frequency AS frequency1, b.frequency AS frequency2
    FROM cause_stats a
    CROSS JOIN cause_stats b
    WHERE a.accident_cause < b.accident_cause
),
pair_calculations AS (
    SELECT cause1, cause2, avg_severity1, avg_severity2, frequency1, frequency2,
        (SELECT AVG(CAST(avg_severity1 AS FLOAT)) FROM cause_pairs) AS overall_avg_severity1,
        (SELECT AVG(CAST(avg_severity2 AS FLOAT)) FROM cause_pairs) AS overall_avg_severity2,
        (SELECT STDEV(CAST(avg_severity1 AS FLOAT)) FROM cause_pairs) AS overall_stdev_severity1,
        (SELECT STDEV(CAST(avg_severity2 AS FLOAT)) FROM cause_pairs) AS overall_stdev_severity2
    FROM cause_pairs
)
SELECT cause1, cause2,
    CASE
        WHEN overall_stdev_severity1 = 0 OR overall_stdev_severity2 = 0 THEN NULL
        ELSE ((avg_severity1 - overall_avg_severity1) * (avg_severity2 - overall_avg_severity2) / 
            (overall_stdev_severity1 * overall_stdev_severity2))
    END AS severity_correlation
FROM pair_calculations
ORDER BY 
 CASE
        WHEN overall_stdev_severity1 = 0 OR overall_stdev_severity2 = 0 THEN 0
        ELSE ABS((avg_severity1 - overall_avg_severity1) * (avg_severity2 - overall_avg_severity2) / 
            (overall_stdev_severity1 * overall_stdev_severity2))
    END DESC;