SELECT
    chart_date,
    rank,
    song_title,
    artist_name,
    last_week,
    peak_position,
    weeks_on_chart,
    EXTRACT(YEAR FROM chart_date) AS chart_year,
    EXTRACT(MONTH FROM chart_date) AS chart_month,
    EXTRACT(WEEK FROM chart_date) AS chart_week,
    FORMAT_DATE('%B', chart_date) AS chart_month_name,
    CASE
        WHEN rank <= 10 THEN 'Top 10'
        WHEN rank <= 25 THEN 'Top 25'
        WHEN rank <= 50 THEN 'Top 50'
        ELSE 'Top 100'
    END AS rank_category
FROM {{ ref('stg_billboard_hot100') }}