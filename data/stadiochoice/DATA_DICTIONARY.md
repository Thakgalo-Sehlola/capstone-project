# Data dictionary v1.0 — STADIOchoice subscriber churn extract

Section 8 of the request asked for this documentation. Everything below is authoritative.

## Official churn definition

> A subscriber has **churned** when their subscription is cancelled and they have **not
> reactivated within 30 days** of the subscription end date.

How the edge cases are treated:

| Case | Treatment |
|---|---|
| **Temporary suspension** | Not churn. `subscription_status = Suspended` covers accounts paused by STADIOchoice, usually for payment reasons. The subscription is still live. |
| **Voluntary pause** | Not churn. A subscriber who accepts a "Pause Subscription" retention offer has `offer_outcome = Paused` and `churn_indicator = False`. |
| **Failed payments** | Not churn on their own. Repeated failure leads to `subscription_status = Lapsed`, which is an involuntary ending and is **excluded** from the churn definition. Only subscriber-initiated cancellation counts. |
| **Reactivation within 30 days** | Not churn. `reactivation_indicator = True`, `churn_indicator = False`, `offer_outcome = Reactivated Later`. |
| **Reactivation after 30 days** | Churn. The original cancellation still counts; the return is a new acquisition. |
| **Retention offer accepted** | Not churn. `offer_accepted = True`, `subscription_end_date` is null, `churn_indicator = False`. |

## Observation period and extraction

- **Observation period: 2024-01-01 to 2026-06-30** (complete historical period covered).
- **Data extraction date: 2026-09-01 08:00:00.** The gap between the observation end and the
  extraction date is deliberate — it lets the 30-day reactivation rule resolve for every
  cancellation in the period.
- **Timezone: Africa/Johannesburg (UTC+02:00)** for every timestamp in every file. South Africa
  does not observe daylight saving.

## Missing-value convention

Missing values are **empty fields**, consistently, in every file. `NULL`, `Unknown`, `-1` and `0`
are never used as placeholders. Blanks that carry meaning:

| Field | Blank means |
|---|---|
| `subscription_end_date` | the subscription is still active |
| `cancellation_date`, `reactivation_date` | did not occur |
| `retention_interaction_date`, `offer_accepted` | no retention offer was made |
| `search_term`, `search_result_count`, `content_click_indicator` | the event was not a Search |
| `province` | not collected at sign-up (~4%) |
| `acquisition_campaign` | organic acquisition, no campaign attached |
| `promotion_code` | no promotion applied |

## Units

| Field | Unit |
|---|---|
| `monthly_fee`, `discount_amount`, `amount_charged`, `refund_amount` | **ZAR** |
| `content_duration`, `viewing_duration` | **seconds** |
| `completion_rate` | proportion, 0-1 |
| all counts | whole numbers |

## Categorical definitions

**`subscription_status`** — Active = currently subscribed and billing normally ·
Cancelled = subscriber-initiated ending · Suspended = temporarily stopped, usually for payment
reasons, subscription still live · Lapsed = ended involuntarily after repeated payment failure ·
Reactivated = returned after a previous ending.

**`offer_outcome`** — Retained = accepted the offer and stayed on the same plan ·
Downgraded = stayed but moved to a cheaper plan · Paused = subscription paused rather than
cancelled · Churned = declined or no offer, and left · Reactivated Later = left and returned.

**`acquisition_channel`** — Digital = online self-service sign-up · Retail Partner = signed up
in a partner store · Call Centre = inbound or outbound telephone sale · Referral = existing
subscriber referral · Telco Bundle = bundled with a mobile or fibre package ·
Field Sales = door-to-door agent.

**`customer_segment`** — STADIOchoice's own segmentation: Mass, Value, Premium,
Sports Enthusiast, Family, At Risk.

**`content_category`** — Live Sport, Drama Series, Comedy, Documentary, Movie, Reality, Kids,
News, Local Drama, Crime.

**`event_type`** — Search, Play, Pause, Scroll, Browse, Add to List, Resume, Stop, Rate.

**`payment_status`** — Paid, Failed, Pending, Refunded.

**`cancellation_reason`** — Price, Content, Technical, Competitor, Not Using, Financial,
Season Ended, Moving, Not Given.

## Key relationships

| Relationship | Cardinality |
|---|---|
| `2_subscriber_account.subscriber_id` → `3_subscription.subscriber_id` | 1 : 1 in this extract |
| `subscriber_id` → `4_viewing_behaviour` | 1 : many |
| `subscriber_id` → `5_engagement_behaviour` | 1 : many |
| `subscriber_id` → `6_billing_payment_history` | 1 : many |
| `subscriber_id` → `7_cancellation_retention` | 1 : 0 or 1 |
| `4_viewing_behaviour.content_id` → `5b_content_reference.content_id` | many : 1 |
| `household_id` | groups subscribers; a household may hold more than one subscription |

Primary keys: `subscriber_id` (tables 2, 3, 7), `subscription_id` (table 3),
`session_id` (table 4), `content_id` (table 5b).

## Known data issues, system changes and coverage limits

1. **Viewing events are missing for roughly 2% of subscribers**, concentrated in the early part
   of the observation period before the analytics SDK was rolled out to all devices. Those
   subscribers appear in tables 2, 3 and 6 but have no rows in table 4.
2. **`completion_rate` for live sport** is measured against scheduled event duration, not actual
   broadcast length, so values slightly above the true proportion are possible for events that
   ran short.
3. **`payment_failure_count`** is the subscriber's total over the observation period, repeated on
   every billing row. It is not a per-event count.
4. **Household identifiers** are reused where more than one subscription belongs to the same
   household, so `household_id` is not unique in table 2.
5. **Search logging** captures the query as typed. Casing and spacing are inconsistent, and
   personal information has been removed where it appeared.
