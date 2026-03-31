SELECT
    'reddit'                             AS platform,
    post_id                              AS content_id,
    subreddit                            AS channel,
    title,
    created_at                           AS published_at,
    score                                AS engagement_score,
    num_comments                         AS comment_count,
    NULL                                 AS view_count,
    NULL                                 AS like_count,
    SAFE_DIVIDE(score, score + 1)        AS engagement_ratio,
    DATE(created_at)                     AS published_date,
    EXTRACT(DAYOFWEEK FROM created_at)   AS day_of_week,
    EXTRACT(HOUR FROM created_at)        AS hour_of_day
FROM {{ ref('stg_reddit_posts') }}