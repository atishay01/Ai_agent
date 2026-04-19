-- =====================================================================
-- 01. Schema overview — run these to show the interviewer the data model
-- =====================================================================

-- List all tables with row counts
SELECT
    relname                          AS table_name,
    to_char(n_live_tup, 'FM999,999,999') AS row_count
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;

-- Show foreign key relationships (the 3NF proof)
SELECT
    tc.table_name    AS child_table,
    kcu.column_name  AS child_column,
    ccu.table_name   AS parent_table,
    ccu.column_name  AS parent_column
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage ccu
  ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
  AND tc.table_schema = 'public'
ORDER BY tc.table_name, kcu.column_name;

-- All indexes on our tables
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
