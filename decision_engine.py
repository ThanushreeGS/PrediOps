"""
decision_engine.py
------------------
AIOps Decision & Root Cause Diagnostic Engine.

Produces for every rack/resource:
1. Root Cause Analysis:
   - Root cause classification (Workload Surge, Policy Shift, Thermal Coupling, Organic Creep, Spike)
   - Primary driver (tenant or facility) and exact share of growth
   - Chronological regime change evidence (date, slope delta, statistical confidence)
   - Risk assessment summary
2. Prescriptive Recommended Decision:
   - Actionable imperative title
   - Concrete, step-by-step operator instructions (with exact rack, tenant, and workload guidance)
   - Action deadline countdown (urgency window)
   - Quantified impact (estimated runway extension in days)
   - Alternative fallback action (Plan B)
   - Interactive operator runbook checklist
"""

from datetime import datetime, timedelta
import pandas as pd


def analyze_cause_and_decision(rack_id: str, resource: str, unit: str,
                               limit: float, safety_margin: float,
                               forecast: dict, change_points: list,
                               attribution: list, current_value: float) -> dict:
    fc = forecast
    is_ok = fc.get("status") == "ok"
    margin_eta = fc.get("to_safety_margin", {}).get("eta_days") if is_ok else None
    margin_date_str = fc.get("to_safety_margin", {}).get("eta_date") if is_ok else None
    range_days = fc.get("to_safety_margin", {}).get("range_days") if is_ok else None
    as_of_str = fc.get("as_of_date", str(datetime.now().date()))
    as_of_date = pd.Timestamp(as_of_str)

    # 1. Determine Urgency Level
    if margin_eta is None:
        urgency_code = "HEALTHY"
        urgency_label = "Healthy / Stable"
        badge_class = "badge-green"
        urgency_score = 3
    elif margin_eta <= 30:
        urgency_code = "URGENT"
        urgency_label = "CRITICAL (Action Required)"
        badge_class = "badge-red"
        urgency_score = 1
    elif margin_eta <= 90:
        urgency_code = "WATCH"
        urgency_label = "WATCH (Planning Horizon)"
        badge_class = "badge-amber"
        urgency_score = 2
    else:
        urgency_code = "HEALTHY"
        urgency_label = "Healthy (> 90 Days)"
        badge_class = "badge-green"
        urgency_score = 3

    top_tenant = attribution[0] if attribution else None
    top_tenant_id = top_tenant["tenant_id"] if top_tenant else ("facility (shared load)" if resource == "thermal_c" else "baseline organic")
    top_share = top_tenant["share_of_rack_growth"] if top_tenant else 0.0
    top_slope = top_tenant["slope_per_day"] if top_tenant else 0.0

    # 2. Determine Cause Analysis
    latest_cp = max(change_points, key=lambda c: c["day_index"]) if change_points else None
    
    if resource == "thermal_c":
        cause_category = "Physical Coupling"
        cause_title = "CRAC Airflow & Cumulative Heat Dissipation"
        if latest_cp:
            cause_summary = (
                f"Intake air temperature experienced a persistent shift on {latest_cp['date']} "
                f"(rate increased from {latest_cp['old_slope']:.4f}°C/day to {latest_cp['new_slope']:.4f}°C/day). "
                f"Thermal load is driven by cumulative compute and power draw across co-located chassis."
            )
        else:
            cause_summary = (
                f"Ambient temperature is tracking sustained electrical dissipation across {rack_id}. "
                f"Current intake: {current_value:.1f}°C (Safety threshold: {safety_margin:.1f}°C)."
            )
    elif latest_cp and top_share >= 0.40:
        old_s = max(1e-5, abs(latest_cp["old_slope"]))
        new_s = latest_cp["new_slope"]
        mult = round(new_s / old_s, 1) if old_s > 0 else 2.0
        cause_category = "Workload Scaling Surge"
        cause_title = f"Tenant Workload Scaling ({top_tenant_id})"
        cause_summary = (
            f"Regime shift detected on {latest_cp['date']}: Growth slope accelerated {mult}× "
            f"(from {latest_cp['old_slope']:+.3f} to {latest_cp['new_slope']:+.3f}/day with {int(latest_cp['confidence']*100)}% confidence). "
            f"Primary cause is tenant '{top_tenant_id}' commanding {int(top_share * 100)}% of net rack growth."
        )
    elif latest_cp:
        cause_category = "Policy / Config Change"
        cause_title = "Infrastructure Policy Modification"
        cause_summary = (
            f"System detected a structural trend shift on {latest_cp['date']}. "
            f"Daily consumption rate altered from {latest_cp['old_slope']:.3f} to {latest_cp['new_slope']:.3f}/day. "
            f"Growth is distributed across multiple tenants sharing {rack_id}."
        )
    elif margin_eta is not None and margin_eta <= 90:
        cause_category = "Organic Buffer Depletion"
        cause_title = "Cumulative Baseline Growth"
        cause_summary = (
            f"Steady organic demand has steadily consumed available safety buffer. "
            f"Current utilization is {current_value:.1f} {unit} against {safety_margin:.1f} {unit} safety margin "
            f"({safety_margin - current_value:.1f} {unit} buffer remaining)."
        )
    else:
        cause_category = "Controlled Baseline"
        cause_title = "Normal Operational State"
        cause_summary = (
            f"Telemetry exhibits stable baseline behavior ({current_value:.1f} {unit}). "
            f"Daily growth rate is minimal ({fc.get('daily_growth_rate', 0.0):.4f} {unit}/day). "
            f"No capacity exhaustion danger detected."
        )

    # 3. Formulate Recommended Decision
    action_deadline_days = (None if margin_eta is None else (0 if margin_eta <= 0 else max(3, int(margin_eta * 0.45))))
    action_deadline_date = (as_of_date + timedelta(days=action_deadline_days)).date().isoformat() if action_deadline_days is not None else "N/A"
    runway_gain_est = int(margin_eta * 2.5) if margin_eta else 180

    if resource == "storage_pct":
        if urgency_code in ("URGENT", "WATCH"):
            decision_title = f"Migrate Tenant Storage & Trim Inactive Volumes"
            decision_action = (
                f"Initiate non-disruptive migration of {int(top_share * 35)}% of block storage volumes "
                f"allocated to '{top_tenant_id}' from {rack_id} to an adjacent rack with spare headroom. "
                f"Execute automated deduplication on stale analytics snapshots."
            )
            decision_steps = [
                f"Issue change notice to tenant '{top_tenant_id}' engineering team.",
                f"Provision replication target on rack with spare storage headroom.",
                f"Perform online volume cutover during low-I/O maintenance window.",
                f"Verify daily storage delta on {rack_id} returns below 0.03%/day."
            ]
            expected_impact = f"Reduces growth slope by ~{int(top_share * 70)}%, extending runway by +{runway_gain_est} days."
            alternative_plan = f"Procure & install a 24-bay NVMe storage shelf on {rack_id} (Hardware lead time: 21 business days)."
            checklist = [
                f"Submit Change Request CR-STO-{rack_id}",
                f"Confirm target storage tier has >40% free space",
                f"Execute storage vMotion / volume migration",
                f"Verify new capacity metrics in next telemetry cycle"
            ]
        else:
            decision_title = "Maintain Standard Tiering & Capacity Hygiene"
            decision_action = "Storage growth is within baseline limits. Continue automated monthly snapshot cleanup."
            decision_steps = ["Review snapshot retention policy.", "Ensure quarterly capacity review."]
            expected_impact = "Sustained headroom > 90 days."
            alternative_plan = "N/A - System healthy."
            checklist = ["Audit orphaned volume reports"]

    elif resource == "power_kw":
        if urgency_code in ("URGENT", "WATCH"):
            decision_title = f"Workload Phase Rebalancing & Dynamic Power Capping"
            decision_action = (
                f"Rebalance high-density power consumers of '{top_tenant_id}' away from {rack_id}. "
                f"Temporarily enforce dynamic power capping (RAPL/BMC) on non-latency-sensitive nodes."
            )
            decision_steps = [
                f"Audit 3-phase PDU draw to identify phase imbalances on {rack_id}.",
                f"Reschedule heavy batch processing jobs to off-peak hours (01:00-06:00 UTC).",
                f"Enforce dynamic power limits to prevent PDU breaker trip thresholds."
            ]
            expected_impact = f"Frees 35-50 kW power headroom, deferring safety threshold by +{runway_gain_est} days."
            alternative_plan = "Request electrical engineering review to rebalance branch circuit loads."
            checklist = [
                f"Log Electrical Work Order for {rack_id}",
                f"Enable power capping profile on chassis BMCs",
                f"Verify PDU current draw stabilizes under 400 kW"
            ]
        else:
            decision_title = "Standard PDU Monitoring & Energy Optimization"
            decision_action = "Power distribution is operating comfortably below safety thresholds."
            decision_steps = ["Monitor weekly PDU phase balance."]
            expected_impact = "Stable operations."
            alternative_plan = "N/A"
            checklist = ["Routine phase balance check"]

    elif resource == "thermal_c":
        if urgency_code in ("URGENT", "WATCH"):
            decision_title = f"Adjust CRAC Dampers & Seal Containment Blanking Panels"
            decision_action = (
                f"Dispatch facility technician to {rack_id} aisle: Increase perforated tile airflow CFM by +15% "
                f"and ensure 100% blanking panel seal across unused rack U-slots to eliminate hot air recirculation."
            )
            decision_steps = [
                f"Audit cold-aisle containment doors for vacuum seal integrity.",
                f"Install 1U/2U blanking panels in all empty slots on {rack_id}.",
                f"Increase CRAC unit fan speed / adjust floor tile dampers."
            ]
            expected_impact = f"Drops rack intake temperature by 1.8°C - 2.5°C, eliminating thermal throttling risk."
            alternative_plan = "Lower building chilled water loop supply temperature by 1.0°C."
            checklist = [
                f"Generate Facilities Ticket: HVAC-{rack_id}",
                f"Inspect airflow tiles with handheld anemometer",
                f"Verify intake temperature drops below 26.0°C"
            ]
        else:
            decision_title = "Routine Thermal Inspection"
            decision_action = "Ambient rack temperature is well within ASHRAE thermal envelope."
            decision_steps = ["Keep cold aisle containment doors closed."]
            expected_impact = "Compliant thermal operations."
            alternative_plan = "N/A"
            checklist = ["Monthly thermal sensor validation"]

    elif resource == "network_mbps":
        if urgency_code in ("URGENT", "WATCH"):
            decision_title = f"Traffic Shaping & LACP Uplink Expansion"
            decision_action = (
                f"Configure QoS rate limits on egress traffic for tenant '{top_tenant_id}' "
                f"and bond secondary 10GbE uplink into the Top-of-Rack (ToR) switch LACP group."
            )
            decision_steps = [
                f"Engage tenant '{top_tenant_id}' to enable compression on bulk egress transfers.",
                f"Provision optical transceiver on second ToR switch port.",
                f"Add link to existing port-channel trunk."
            ]
            expected_impact = f"Doubles uplink throughput to 20 Gbps, pushing exhaustion horizon past +{runway_gain_est} days."
            alternative_plan = "Offload non-production backups over dedicated out-of-band management interface."
            checklist = [
                f"Create Network CR: NET-{rack_id}-TRUNK",
                f"Stage optic in switch port ge-0/0/2",
                f"Validate LACP bundle bandwidth"
            ]
        else:
            decision_title = "Normal Bandwidth Surveillance"
            decision_action = "Network interface utilization is healthy with plenty of burst headroom."
            decision_steps = ["Review weekly port traffic statistics."]
            expected_impact = "No packet drops or queuing delays."
            alternative_plan = "N/A"
            checklist = ["Verify port error counters"]

    elif resource == "vm_count":
        if urgency_code in ("URGENT", "WATCH"):
            decision_title = f"Live VM Migration (vMotion) to Spare Capacity Nodes"
            decision_action = (
                f"Execute automated live migration of 20-35 VMs belonging to '{top_tenant_id}' "
                f"from {rack_id} to an under-allocated destination rack."
            )
            decision_steps = [
                f"Identify idle/low-priority development instances for consolidation.",
                f"Run DRS / scheduler rebalance to live-migrate selected VMs.",
                f"Verify zero service interruption for tenant applications."
            ]
            expected_impact = f"Relieves rack hypervisor density below 340 VMs, unlocking +{runway_gain_est} days runway."
            alternative_plan = "Add additional compute blade to chassis or review memory overcommit ratios."
            checklist = [
                f"Initiate VM Orchestration Task VM-MIG-{rack_id}",
                f"Validate hypervisor cluster compatibility",
                f"Execute live vMotion without downtime",
                f"Confirm rack VM count stabilizes below safety threshold"
            ]
        else:
            decision_title = "Standard Hypervisor Load Balancing"
            decision_action = "VM density is well within safe hypervisor overcommit boundaries."
            decision_steps = ["Maintain standard automated DRS balancing."]
            expected_impact = "Optimal compute utilization."
            alternative_plan = "N/A"
            checklist = ["Weekly hypervisor health audit"]

    else:
        decision_title = "Monitor Resource Trend"
        decision_action = f"Maintain ongoing monitoring of {resource} on {rack_id}."
        decision_steps = ["Inspect next reporting cycle."]
        expected_impact = "Normal operations."
        alternative_plan = "N/A"
        checklist = ["Log observation"]

    return {
        "urgency": {
            "code": urgency_code,
            "label": urgency_label,
            "badge_class": badge_class,
            "score": urgency_score,
            "margin_eta_days": margin_eta,
            "margin_eta_date": margin_date_str,
            "range_days": range_days,
        },
        "cause_analysis": {
            "category": cause_category,
            "title": cause_title,
            "summary": cause_summary,
            "primary_driver": top_tenant_id,
            "driver_share_pct": round(top_share * 100, 1),
            "driver_slope": round(top_slope, 4),
            "regime_shift_detected": bool(latest_cp),
            "latest_regime_change": latest_cp,
            "current_utilization": round(current_value, 2),
            "safety_margin": safety_margin,
            "hard_limit": limit,
            "unit": unit,
        },
        "recommended_decision": {
            "title": decision_title,
            "action_summary": decision_action,
            "prescriptive_steps": decision_steps,
            "action_deadline_days": action_deadline_days,
            "action_deadline_date": action_deadline_date,
            "expected_runway_gain_days": runway_gain_est,
            "expected_impact": expected_impact,
            "alternative_fallback": alternative_plan,
            "operator_checklist": checklist,
        }
    }
