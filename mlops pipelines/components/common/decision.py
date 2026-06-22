def derive_decision(p_deg, p_crit, risk_score):
    if risk_score < 0.3:
        risk_level = "low"
    elif risk_score < 0.6:
        risk_level = "medium"
    else:
        risk_level = "high"

    if p_crit > 0.7:
        recommended_action = "dispatch_technician"
    elif p_deg > 0.7:
        recommended_action = "monitor"
    else:
        recommended_action = "no_action"

    if risk_score > 0.8:
        priority_level = 1
    elif risk_score > 0.5:
        priority_level = 2
    else:
        priority_level = 3

    return risk_level, recommended_action, priority_level