"""Deterministic offline template fallback per predicted class and unknown_anomaly flag combination.

Built strictly from verbatim packet numbers, labelled source `"template"`, and guaranteed to pass
every deterministic check in `ExplanationGuard`.
"""
from __future__ import annotations

from .packet import ExplainerPacket


def generate_template_explanation(packet: ExplainerPacket) -> dict[str, str]:
    """Generate a deterministic 2-to-3 sentence summary and 1 sentence suggested check
    using ONLY numbers in `packet` and naming ONLY `packet.predicted_class`."""
    cls = packet.predicted_class
    t0 = packet.time_s.t_start
    t1 = packet.time_s.t_end
    score = packet.model_score
    t_n1 = packet.node_mean_tilt_deg.NODE_01
    t_n2 = packet.node_mean_tilt_deg.NODE_02
    t_n3 = packet.node_mean_tilt_deg.NODE_03
    t_n4 = packet.node_mean_tilt_deg.NODE_04
    p_net = packet.piezo_rms.network_mean
    l_cm = packet.lidar_range_change_m.hub_common_mode_m
    l_n1 = packet.lidar_range_change_m.NODE_01
    c_miss = packet.comm.missing_fraction_network
    c_lat = packet.comm.latency_max_s
    res_m = packet.radial_convergence_residual_m
    w_cos = packet.radial_convergence_weighted_cos
    anom = packet.anomaly_score

    if packet.unknown_anomaly:
        flag_sentence = (
            f"The advisory anomaly detector is flagged with anomaly_score {anom} "
            f"(radial convergence residual {res_m} m and weighted cosine {w_cos})."
        )
    else:
        flag_sentence = (
            f"The advisory anomaly detector is unflagged with anomaly_score {anom} "
            f"(radial convergence residual {res_m} m and weighted cosine {w_cos})."
        )

    if cls == "gradual":
        s1 = (
            f"In this simulated window from {t0} s to {t1} s, the frozen classifier output is consistent with "
            f"gradual movement at model_score {score}."
        )
        s2 = (
            f"Simulated node mean tilts are {t_n1} deg on NODE_01, {t_n2} deg on NODE_02, "
            f"{t_n3} deg on NODE_03, and {t_n4} deg on NODE_04, with LiDAR range change {l_n1} m on NODE_01."
        )
        check = (
            f"Inspect simulated multi-node tilt progression across NODE_01 ({t_n1} deg) through NODE_04 ({t_n4} deg) "
            f"and radial convergence residual ({res_m} m)."
        )
    elif cls == "rapid":
        s1 = (
            f"In this simulated window from {t0} s to {t1} s, the frozen classifier output is consistent with "
            f"rapid movement at model_score {score}."
        )
        s2 = (
            f"Simulated node mean tilts reach {t_n1} deg on NODE_01, {t_n2} deg on NODE_02, "
            f"{t_n3} deg on NODE_03, and {t_n4} deg on NODE_04, with network piezo RMS {p_net} V."
        )
        check = (
            f"Compare simulated node tilt rates across NODE_01 ({t_n1} deg) to NODE_04 ({t_n4} deg) "
            f"against LiDAR hub common-mode shift ({l_cm} m)."
        )
    elif cls == "abnormal_vibration":
        s1 = (
            f"In this simulated window from {t0} s to {t1} s, the frozen classifier output is consistent with "
            f"abnormal_vibration at model_score {score}."
        )
        s2 = (
            f"Simulated network mean piezo RMS is {p_net} V (NODE_01 piezo RMS {packet.piezo_rms.NODE_01} V, "
            f"NODE_02 {packet.piezo_rms.NODE_02} V) while NODE_01 mean tilt is {t_n1} deg."
        )
        check = (
            f"Review simulated piezo acoustic channels across the four nodes relative to network mean piezo RMS ({p_net} V)."
        )
    elif cls == "sensor_fault":
        s1 = (
            f"In this simulated window from {t0} s to {t1} s, the frozen classifier output is consistent with "
            f"sensor_fault at model_score {score}."
        )
        s2 = (
            f"Simulated node mean tilts are {t_n1} deg on NODE_01, {t_n2} deg on NODE_02, "
            f"{t_n3} deg on NODE_03, and {t_n4} deg on NODE_04, with LiDAR hub common-mode change {l_cm} m."
        )
        check = (
            f"Check simulated transducer health on NODE_01 ({t_n1} deg) through NODE_04 ({t_n4} deg) "
            f"and LiDAR hub common-mode offset ({l_cm} m)."
        )
    elif cls == "comm_fault":
        s1 = (
            f"In this simulated window from {t0} s to {t1} s, the frozen classifier output is consistent with "
            f"comm_fault at model_score {score}."
        )
        s2 = (
            f"Simulated network missing packet fraction is {c_miss} with maximum latency {c_lat} s "
            f"and mean latency {packet.comm.latency_mean_s} s."
        )
        check = (
            f"Check simulated LoRa link reception logs for missing packet fraction ({c_miss}) and maximum latency ({c_lat} s)."
        )
    else:  # normal
        s1 = (
            f"In this simulated window from {t0} s to {t1} s, the frozen classifier output is consistent with "
            f"normal baseline conditions at model_score {score}."
        )
        s2 = (
            f"Simulated node mean tilts are {t_n1} deg on NODE_01, {t_n2} deg on NODE_02, "
            f"{t_n3} deg on NODE_03, and {t_n4} deg on NODE_04, with network piezo RMS {p_net} V and missing packet fraction {c_miss}."
        )
        check = (
            f"Continue routine monitoring of simulated telemetry channels across NODE_01 ({t_n1} deg) through NODE_04 ({t_n4} deg)."
        )

    return {
        "summary": f"{s1} {s2} {flag_sentence}",
        "suggested_check": check,
    }
