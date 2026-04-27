-- =====================================================================
-- Read-only role for the LangChain SQL agent.
--
-- The agent and dashboard connect with this role instead of the admin
-- ``postgres`` user. Even if the application-level SQL guardrail had a
-- bug tomorrow, the database itself would refuse any write — defence
-- in depth.
--
-- Run once after the schema and data are loaded:
--
--   psql -U postgres -d olist_db -f src/seed_readonly_user.sql
--
-- The role's password must match ``PG_PASSWORD_AGENT`` in your .env.
-- Edit the value below before running, OR pass it via psql variable:
--
--   psql -U postgres -d olist_db \
--        -v agent_password="'change_me_to_a_strong_value'" \
--        -f src/seed_readonly_user.sql
-- =====================================================================

-- Drop existing role cleanly so the script is idempotent.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'olist_agent') THEN
        REVOKE ALL ON ALL TABLES IN SCHEMA public FROM olist_agent;
        REVOKE ALL ON SCHEMA public FROM olist_agent;
        DROP ROLE olist_agent;
    END IF;
END
$$;

-- Edit this password (or override via -v agent_password=...).
CREATE ROLE olist_agent
    WITH LOGIN PASSWORD :'agent_password'
         NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION
         CONNECTION LIMIT 20;

-- Schema-level: just enough to see what's there.
GRANT USAGE ON SCHEMA public TO olist_agent;

-- Table-level: SELECT only on the 8 warehouse tables.
GRANT SELECT ON
    customers, sellers, products, geolocation,
    orders, order_items, order_payments, order_reviews
TO olist_agent;

-- Future tables (e.g. mart_* views from commit 2) inherit SELECT.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO olist_agent;

-- Belt and braces: explicitly REVOKE the three things we never want.
-- (NOSUPERUSER + GRANT-only-SELECT already implies these, but being
-- explicit makes audits easy.)
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON
    customers, sellers, products, geolocation,
    orders, order_items, order_payments, order_reviews
FROM olist_agent;

REVOKE CREATE ON SCHEMA public FROM olist_agent;

-- Sanity-check: confirm the role can SELECT and only SELECT.
\echo
\echo 'Privileges granted to olist_agent:'
SELECT grantee, table_name, privilege_type
FROM information_schema.role_table_grants
WHERE grantee = 'olist_agent'
ORDER BY table_name, privilege_type;
