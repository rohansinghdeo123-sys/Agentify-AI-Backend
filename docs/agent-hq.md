# Agent HQ

Agent HQ opens from the Admin landing page at `/dashboard/internal/admin/agent-hq`. Choose Chat or Report on an agent's portrait. Admin Diagnostics opens the existing evidence console. The five HQ views use the existing platform registry, coach workflow records, dispatch events, model/tool traces and runtime handoffs. The organization distinguishes six registered platform agents from ten roles inside the coach workflow. Portraits represent fictional AI roles.

## Executive access

Every HQ endpoint verifies a Firebase token and requires an existing backend administrator. Rohan Singh Deo and Amit have the same permissions. By default, HQ matches each existing product-owner email against an email-verified token. Set `AGENT_HQ_ROHAN_UID` and `AGENT_HQ_AMIT_UID` to bind the corresponding roles to exact Firebase UIDs. The pinned accounts still need the existing backend administrator entitlement.

Requests and executive decisions are written to the existing admin audit log. Both executives can see the shared operational conversation history. Use the request field for operational questions and keep student details and credentials out of it. Trace views expose allowlisted metadata and omit student content, agent prompts, tool arguments, and raw exceptions.

## Operation

Run `python scripts/migrate.py` before starting the backend. Revision `20261117_0020` adds executive runs, ordered messages, recommendation decisions, and the standup schedule. The existing background queue picks up durable jobs. The schedule is off by default. Enable it in Agent HQ and choose a local time and IANA timezone. The default is 09:00 Asia/Kolkata with in-app delivery. Reports support 1–90 days and an optional UTC closing date.

Meeting and standup jobs collect role-specific findings from one sanitized evidence snapshot. They persist the delegation and each participant's contributions, then record the consolidated summary. In direct chat, greetings work locally, and other general questions use the existing Groq configuration with up to three previous turns. Report and performance questions use measured evidence. When enabled with `AGENT_HQ_MODEL_ANALYSIS=true`, operational conversations and meetings can add a bounded model interpretation labeled as analysis. Reports and measured findings remain available when the provider fails. Listen uses the browser's speech synthesis when supported. The portrait animates during active work and shows a waveform during playback.

The API allows up to 40 manual jobs per executive per rolling day, two concurrent pending jobs per executive, and 16 roles per job. Jobs use idempotent request IDs. The queue uses a two-minute execution lease and marks interrupted jobs failed. Submit a new request to retry. The server sends authenticated event-stream invalidations and retains results for reconnect and refresh. Historical run lists support cursor pagination.

Accepting, deferring, or dismissing a recommendation records its disposition. The available tasks inspect telemetry, generate reports, and organize findings. The service has no deployment, shell, production configuration, database mutation, or student-data export tools.

## Measurement coverage

Counts and latency derive from recorded operations. Percentiles use a bounded recent sample, with the sample size visible in coverage. Token and cost figures are estimates when source traces include them. Exact billing, first-token timing, queue latency, and host resource utilization require additional telemetry. Runtime status reflects recent observations. Missing observations show as unknown. Executive requests and their messages are persisted independently of student activity metrics.

## Portrait assets

The built-in image generator produced two contact sheets for use as WebP sprites at `public/agent-hq/portraits.webp` and `public/agent-hq/workflow-portraits.webp` in the frontend. The prompt specified evenly aligned 4×3 and 2×2 sheets of distinct fictional professionals, with centered editorial headshots, tailored clothing, soft natural light, and muted backgrounds.
